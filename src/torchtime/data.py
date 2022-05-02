import csv
import os
import pathlib
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sktime.datasets import load_UCR_UEA_dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from torchtime.constants import EPS
from torchtime.impute import forward_impute, replace_missing
from torchtime.utils import (
    download_file,
    get_file_list,
    nan_mode,
    physionet_download,
    print_message,
    sample_indices,
)


class _TimeSeriesDataset(Dataset):
    """**Generic time series PyTorch Dataset.**

    .. warning::
        Overload the ``_get_data()`` method to define a data set.

    The proportion of data in the training, validation and (optional) test data sets are
    specified by the ``train_prop`` and ``val_prop`` arguments. For a
    training/validation split specify ``train_prop`` only. For a
    training/validation/test split specify both ``train_prop`` and ``val_prop``. For
    example ``train_prop=0.8`` generates a 80/20% train/validation split, but
    ``train_prop=0.8``, ``val_prop=0.1`` generates a 80/10/10% train/validation/test
    split. Splits are formed using stratified sampling.

    The ``split`` argument determines which data set is returned.

    Missing data can be simulated using the ``missing`` argument. Data are dropped at
    random.

    Missing data are imputed using the ``impute`` argument. *mean* imputation replaces
    missing values with the training data channel mean. *forward* imputation replaces
    missing values with the previous observation. Alternatively a custom imputation
    function can be passed to ``impute``. This must accept ``X`` (raw time series),
    ``y`` (labels) and ``fill`` (channel means/modes) tensors and return ``X`` and ``y``
    tensors post imputation.

    .. warning::
        Mean and forward imputation are unsuitable for categorical variables. To impute
        missing values for a categorical variable with the channel mode (rather than
        the channel mean), pass the channel indices to the ``categorical`` argument.
        Note this is also required for forward imputation to appropriately impute any
        initial missing values.

        Alternatively, the calculated channel mean/mode can be overridden using the
        ``override`` argument. This can be used to impute missing data with a fixed
        value.

    The ``time``, ``mask`` and ``delta`` arguments append additional channels. By
    default, a time stamp is added as the first channel. ``mask`` adds a missing data
    mask and ``delta`` adds the time since the previous observation for each channel.
    Time deltas are calculated as in `Che et al (2018)
    <https://doi.org/10.1038/s41598-018-24271-9>`_.

    Processed data are cached in the ``./.torchtime/[dataset name]`` directory by
    default. The location can be changed with the ``path`` argument, for example to
    share a single cache location across projects.

    .. note::
        When passed to a PyTorch DataLoader, batches are a named dictionary with ``X``,
        ``y`` and ``length`` data.

    Args:
        dataset: Name of cache directory for data set.
        split: The data split to return, either *train*, *val* (validation) or *test*.
        train_prop: Proportion of data in the training set.
        val_prop: Proportion of data in the validation set (optional, see above).
        missing: The proportion of data to drop at random. If ``missing`` is a single
            value, data are dropped from all channels. To drop data independently across
            each channel, pass a list of rates with the proportion missing for each
            channel (default 0 i.e. no missing data simulation).
        impute: Method used to impute missing data, either *none*, *mean*, *forward* or
            a custom imputer function (default "none").
        categorical: List with channel indices of categorical variables (default ``[]``
            i.e. no categorical variables).
        override: Override the calculated channel mean/mode when imputing data.
            Dictionary with channel indices and values e.g. {1: 4.5, 3: 7.2} (default
            ``{}`` i.e. no overridden channel mean/modes).
        time: Append time stamp in the first channel (default True).
        mask: Append missing data mask channels (default False).
        delta: Append time delta channels (default False).
        downscale: The proportion of data to return. Use to reduce the size of the data
            set when testing a model (default 1).
        path: Location of the ``.torchtime`` cache directory (default ".").
        seed: Random seed for reproducibility (optional).

    Attributes:
        X (Tensor): A tensor of default shape (*n*, *s*, *c* + 1) where *n* = number of
            trajectories, *s* = (maximum) trajectory length and *c* = number of
            channels. By default, a time stamp is appended as the first channel. If
            ``time`` is False, the time stamp is omitted and the tensor has shape
            (*n*, *s*, *c*).

            A missing data mask and/or time delta channels can be appended with the
            ``mask`` and ``delta`` arguments. These each have the same number of
            channels as the data set. For example, if ``time``, ``mask`` and
            ``delta`` are all True, ``X`` has shape (*n*, *s*, 3 * *c* + 1) and the
            channels are in the order: time stamp, time series, missing data mask, time
            deltas.

            Where trajectories are of unequal lengths they are padded with ``NaNs`` to
            the length of the longest trajectory in the data.
        y (Tensor): One-hot encoded label data. A tensor of shape (*n*, *l*) where *l*
            is the number of classes.
        length (Tensor): Length of each trajectory prior to padding. A tensor of shape
            (*n*).

    .. note::
        ``X``, ``y`` and ``length`` are available for the training, validation and test
        splits by appending ``_train``, ``_val`` and ``_test`` respectively. For
        example, ``y_val`` returns the labels for the validation data set. These
        attributes are available regardless of the ``split`` argument.

    Returns:
        torch.utils.data.Dataset: A PyTorch Dataset object which can be passed to a
        DataLoader.
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        train_prop: float,
        val_prop: float = None,
        missing: Union[float, List[float]] = 0,
        impute: Union[str, Callable[[Tensor], Tensor]] = "none",
        categorical: List[int] = [],
        override: Dict[int, float] = {},
        time: bool = True,
        mask: bool = False,
        delta: bool = False,
        downscale: float = 1.0,
        path: str = ".",
        seed: int = None,
    ) -> None:
        """
        Data processing pipeline:

            1. Download data and prepare X, y, length tensors, downsampling if required
            3. Simulate missing data
            4. Add time/missing data mask/time delta channels
            5. Split into training/validation/test data sets
            6. Impute missing data
            7. Assign ``split`` to X, y, length attributes
        """
        self.val_prop = val_prop
        self.missing = missing
        self.categorical = categorical
        self.time = time
        self.mask = mask
        self.downscale = downscale
        self.seed = seed
        self.test_prop = 0

        # Constants
        self.PATH = pathlib.Path() / path / ".torchtime" / dataset
        self.IMPUTE_FUNCTIONS = {
            "none": self._no_imputation,
            "mean": self._mean_imputation,
            "forward": self._forward_imputation,
        }

        # Validate imputation arguments
        impute_options = self.IMPUTE_FUNCTIONS.keys()
        impute_error = "argument 'impute' must be a string in {} or a function".format(
            impute_options
        )
        if type(impute) is str:
            assert impute in impute_options, impute_error
            imputer = self.IMPUTE_FUNCTIONS.get(impute)
        elif callable(impute):
            imputer = impute
        else:
            raise Exception(impute_error)
        if impute == "none" and (self.categorical != [] or override != {}):
            print_message(
                "No missing data imputation therefore any 'categorical' or 'override' arguments have been ignored",  # noqa: E501
                type="error",
            )
            self.categorical = []
            override = {}
        assert type(categorical) is list, "argument 'categorical' must be a list"
        assert type(override) is dict, "argument 'override' must be a dictionary"

        # Data splits
        assert (
            train_prop > EPS and train_prop < 1
        ), "argument 'train_prop' must be in range (0, 1)"
        if self.val_prop is None:
            self.val_prop = 1 - train_prop
        else:
            assert (
                self.val_prop > EPS and self.val_prop < 1 - train_prop
            ), "argument 'val_prop' must be in range (0, 1-train_prop)"
            self.test_prop = 1 - train_prop - self.val_prop
            self.val_prop = self.val_prop / (1 - self.test_prop)

        # Validate split argument
        splits = ["train", "val"]
        if self.test_prop > EPS:
            splits.append("test")
        assert split in splits, "argument 'split' must be one of {}".format(splits)

        # 1. If no downscaling, get data from cache (or if no cache, call _get_data()
        # and cache results), otherwise get subset of data and do not cache
        if self._check_cache() and self.downscale >= (1 - EPS):
            X_all, y_all, length_all = self._load_cache()
        else:
            X_all, y_all, length_all = self._get_data()
            X_all = X_all.float()  # float32 precision
            y_all = y_all.float()  # float32 precision
            length_all = length_all.long()  # int64 precision
            if self.downscale >= (1 - EPS):
                self._cache_data(X_all, y_all, length_all)

        # 2. Simulate missing data
        if (type(self.missing) is list and sum(self.missing) > EPS) or (
            type(self.missing) is float and self.missing > EPS
        ):
            self._simulate_missing(X_all)

        # 3. Add time stamp/mask/time delta channels
        if self.time:
            X_all = torch.cat([self._time_stamp(X_all), X_all], dim=2)
        if self.mask:
            X_all = torch.cat([X_all, self._missing_mask(X_all)], dim=2)
        if delta:
            X_all = torch.cat([X_all, self._time_delta(X_all)], dim=2)

        # 4. Form train/validation/test splits
        stratify = torch.nansum(y_all, dim=1) > 0
        (
            self.X_train,
            self.y_train,
            self.length_train,
            self.X_val,
            self.y_val,
            self.length_val,
            self.X_test,
            self.y_test,
            self.length_test,
        ) = self._split_data(
            X_all,
            y_all,
            length_all,
            stratify,
        )

        # 5. Impute missing data (no missing values in time, mask and delta channels)
        fill = torch.nanmean(torch.flatten(self.X_train, end_dim=1), dim=0)
        n_data_channels = int((len(fill) - self.time) / (1 + self.mask + delta) - 1)
        # Impute using mode if categorical variable
        if self.categorical != []:
            self.categorical = [idx + time for idx in categorical]
            assert (
                max(categorical) <= n_data_channels
            ), "indices in 'categorical' should be between 0 and {}".format(
                n_data_channels
            )
            train_modes = [
                nan_mode(self.X_train[:, :, channel]) for channel in self.categorical
            ]
            for i, idx in enumerate(self.categorical):
                fill[idx] = train_modes[i]
        # Override mean/mode if required
        if override != {}:
            for x, y in override.items():
                assert (
                    x <= n_data_channels
                ), "indices in 'override' should be between 0 and {}".format(
                    n_data_channels
                )
                fill[x + self.time] = y
        # Impute data
        self.X_train, self.y_train = imputer(self.X_train, self.y_train, fill)
        self.X_val, self.y_val = imputer(self.X_val, self.y_val, fill)
        if self.test_prop > EPS:
            self.X_test, self.y_test = imputer(self.X_test, self.y_test, fill)
        else:
            del self.X_test, self.y_test, self.length_test

        # 6. Return data split
        if split == "test":
            self.X = self.X_test
            self.y = self.y_test
            self.length = self.length_test
        elif split == "train":
            self.X = self.X_train
            self.y = self.y_train
            self.length = self.length_train
        else:
            self.X = self.X_val
            self.y = self.y_val
            self.length = self.length_val

    @staticmethod
    def _no_imputation(X, y, _):
        """No imputation."""
        return X, y

    @staticmethod
    def _mean_imputation(X, y, fill):
        """Mean imputation. Replace missing values in ``X`` from ``fill``. Replace
        missing values in ``y`` with zeros."""
        X_imputed = replace_missing(X, fill=fill)
        y_imputed = replace_missing(y, fill=torch.zeros(y.size(-1)))
        return X_imputed, y_imputed

    @staticmethod
    def _forward_imputation(X, y, fill):
        """Forward imputation. Replace missing values with previous observation. Replace
        any initial missing values in ``X`` from ``fill``. Assume no missing initial
        values in ``y`` but there may be trailing missing values due to padding."""
        X_imputed = forward_impute(X, fill=fill)
        y_imputed = forward_impute(y)
        return X_imputed, y_imputed

    def _check_cache(self):
        """Check for cached data."""
        return (
            (self.PATH / "X.pt").is_file()
            and (self.PATH / "y.pt").is_file()
            and (self.PATH / "length.pt").is_file()
        )

    def _load_cache(self):
        """Load data from cache."""
        X = torch.load(self.PATH / "X.pt")
        y = torch.load(self.PATH / "y.pt")
        length = torch.load(self.PATH / "length.pt")
        return X, y, length

    def _cache_data(self, X, y, length):
        """Cache tensors."""
        if not self.PATH.is_dir():
            os.makedirs(self.PATH)
        torch.save(X, self.PATH / "X.pt")
        torch.save(y, self.PATH / "y.pt")
        torch.save(length, self.PATH / "length.pt")

    def _get_data(self):
        """Download data and form X, y, length tensors. Overload this function to
        create a data set class."""
        raise NotImplementedError

    def _simulate_missing(self, X):
        """Simulate missing data by modifying X in place."""
        length = X.size(1)
        generator = torch.Generator()
        if self.seed is not None:
            generator = generator.manual_seed(self.seed)
        for Xi in X:
            if type(self.missing) in [int, float]:
                idx = sample_indices(length, self.missing, generator)
                Xi[idx] = float("nan")
            else:
                assert Xi.size(-1) == len(
                    self.missing
                ), "argument 'missing' must be same length as number of channels \
                    ({})".format(
                    Xi.size(-1)
                )
                for channel, rate in enumerate(self.missing):
                    idx = sample_indices(length, rate, generator)
                    Xi[idx, channel] = float("nan")

    @staticmethod
    def _time_stamp(X):
        """Calculate time stamp."""
        time_stamp = torch.arange(X.size(1)).unsqueeze(0)
        time_stamp = time_stamp.tile((X.size(0), 1)).unsqueeze(2)
        return time_stamp

    def _missing_mask(self, X):
        """Calculate missing data mask."""
        mask = torch.logical_not(torch.isnan(X[:, :, self.time :]))
        return mask

    def _time_delta(self, X):
        """Calculate time delta calculated as in Che et al, 2018, see
        https://www.nature.com/articles/s41598-018-24271-9."""
        # Add time and mask channels
        if not self.time:
            X = torch.cat([self._time_stamp(X), X], dim=2)
        if not self.mask:
            X = torch.cat([X, self._missing_mask(X)], dim=2)
        # Time of each observation by channel
        n_channels = int((X.size(-1) - 1) / 2)
        X = X.transpose(1, 2)  # shape (n, c, s)
        time_stamp = X[:, 0].unsqueeze(1).repeat(1, n_channels, 1)
        # Time delta/mask are 0/1 at time 0 by definition
        time_delta = time_stamp.clone()
        time_delta[:, :, 0] = 0
        time_mask = X[:, -n_channels:].clone()
        time_mask[:, :, 0] = 1
        # Time of previous observation if data missing
        time_delta = time_delta.gather(-1, torch.cummax(time_mask, -1)[1])
        # Calculate time delta
        time_delta = torch.cat(
            (
                time_delta[:, :, 0].unsqueeze(2),  # t = 0
                time_stamp[:, :, 1:]
                - time_delta[:, :, :-1],  # i.e. time minus time of previous observation
            ),
            dim=2,
        )
        return time_delta.transpose(1, 2)

    def _split_data(self, X, y, length, stratify):
        """Split data (X, y, length) into training, validation and (optional) test sets
        using stratified sampling."""
        random_state = np.random.RandomState(self.seed)
        if self.test_prop > EPS:
            # Test split
            test_nontest_split = train_test_split(
                X,
                y,
                length,
                stratify,
                train_size=self.test_prop,
                random_state=random_state,
                shuffle=True,
                stratify=stratify,
            )
            (
                X_test,
                X_nontest,
                y_test,
                y_nontest,
                length_test,
                length_nontest,
                _,
                stratify_nontest,
            ) = test_nontest_split
            # Validation/train split
            val_train_split = train_test_split(
                X_nontest,
                y_nontest,
                length_nontest,
                train_size=self.val_prop,
                random_state=random_state,
                shuffle=True,
                stratify=stratify_nontest,
            )
            X_val, X_train, y_val, y_train, length_val, length_train = val_train_split
        else:
            # Validation/train split
            val_train_split = train_test_split(
                X,
                y,
                length,
                train_size=self.val_prop,
                random_state=random_state,
                shuffle=True,
                stratify=stratify,
            )
            X_val, X_train, y_val, y_train, length_val, length_train = val_train_split
            X_test, y_test, length_test = float("nan"), float("nan"), float("nan")
        return (
            X_train,
            y_train,
            length_train,
            X_val,
            y_val,
            length_val,
            X_test,
            y_test,
            length_test,
        )

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx], "length": self.length[idx]}


class TensorTimeSeriesDataset(_TimeSeriesDataset):
    """
    **Returns a PyTorch Dataset from input tensors.**

    Pass ``X``, ``y`` and ``length`` tensors and a ``dataset`` name to create the data
    set.

    The proportion of data in the training, validation and (optional) test data sets are
    specified by the ``train_prop`` and ``val_prop`` arguments. For a
    training/validation split specify ``train_prop`` only. For a
    training/validation/test split specify both ``train_prop`` and ``val_prop``. For
    example ``train_prop=0.8`` generates a 80/20% train/validation split, but
    ``train_prop=0.8``, ``val_prop=0.1`` generates a 80/10/10% train/validation/test
    split. Splits are formed using stratified sampling.

    The ``split`` argument determines which data set is returned.

    Missing data can be simulated using the ``missing`` argument. Data are dropped at
    random.

    Missing data are imputed using the ``impute`` argument. *mean* imputation replaces
    missing values with the training data channel mean. *forward* imputation replaces
    missing values with the previous observation. Alternatively a custom imputation
    function can be passed to ``impute``. This must accept ``X`` (raw time series),
    ``y`` (labels) and ``fill`` (channel means/modes) tensors and return ``X`` and ``y``
    tensors post imputation.

    .. warning::
        Mean and forward imputation are unsuitable for categorical variables. To impute
        missing values for a categorical variable with the channel mode (rather than
        the channel mean), pass the channel indices to the ``categorical`` argument.
        Note this is also required for forward imputation to appropriately impute any
        initial missing values.

        Alternatively, the calculated channel mean/mode can be overridden using the
        ``override`` argument. This can be used to impute missing data with a fixed
        value.

    The ``time``, ``mask`` and ``delta`` arguments append additional channels. By
    default, a time stamp is added as the first channel. ``mask`` adds a missing data
    mask and ``delta`` adds the time since the previous observation for each channel.
    Time deltas are calculated as in `Che et al (2018)
    <https://doi.org/10.1038/s41598-018-24271-9>`_.

    Data are cached in the ``./.torchtime/[dataset name]`` directory by default. The
    location can be changed with the ``path`` argument, for example to share a single
    cache location across projects.

    .. note::
        When passed to a PyTorch DataLoader, batches are a named dictionary with ``X``,
        ``y`` and ``length`` data.

    Args:
        dataset: Name of cache directory for data set.
        X: Tensor with time series data of shape (*n*, *s*, *c*).
        y: Tensor with label data of shape (*n*, *s*, *l*) or (*n*, *s*, *l*).
        length: Tensor with time series lengths of shape (*n*).
        split: The data split to return, either *train*, *val* (validation) or *test*.
        train_prop: Proportion of data in the training set.
        val_prop: Proportion of data in the validation set (optional, see above).
        missing: The proportion of data to drop at random. If ``missing`` is a single
            value, data are dropped from all channels. To drop data independently across
            each channel, pass a list of rates with the proportion missing for each
            channel (default 0 i.e. no missing data simulation).
        impute: Method used to impute missing data, either *none*, *mean*, *forward* or
            a custom imputer function (default "none").
        categorical: List with channel indices of categorical variables (default ``[]``
            i.e. no categorical variables).
        override: Override the calculated channel mean/mode when imputing data.
            Dictionary with channel indices and values e.g. {1: 4.5, 3: 7.2} (default
            ``{}`` i.e. no overridden channel mean/modes).
        time: Append time stamp in the first channel (default True).
        mask: Append missing data mask channels (default False).
        delta: Append time delta channels (default False).
        downscale: The proportion of data to return. Use to reduce the size of the data
            set when testing a model (default 1).
        path: Location of the ``.torchtime`` cache directory (default ".").
        seed: Random seed for reproducibility (optional).

    Attributes:
        X (Tensor): A tensor of default shape (*n*, *s*, *c* + 1) where *n* = number of
            trajectories, *s* = trajectory length and *c* = number of channels. By
            default, a time stamp is appended as the first channel. If ``time`` is
            False, the time stamp is omitted and the tensor has shape (*n*, *s*, *c*).

            A missing data mask and/or time delta channels can be appended with the
            ``mask`` and ``delta`` arguments. These each have the same number of
            channels as the data set. For example, if ``time``, ``mask`` and
            ``delta`` are all True, ``X`` has shape (*n*, *s*, 3 * *c* + 1) and the
            channels are in the order: time stamp, time series, missing data mask, time
            deltas.
        y (Tensor): Label data. Shape equal to input tensor ``y``.
        length (Tensor): Length of each trajectory prior to padding. A tensor of shape
            (*n*).

    .. note::
        ``X``, ``y`` and ``length`` are available for the training, validation and test
        splits by appending ``_train``, ``_val`` and ``_test`` respectively. For
        example, ``y_val`` returns the labels for the validation data set. These
        attributes are available regardless of the ``split`` argument.

    Returns:
        torch.utils.data.Dataset: A PyTorch Dataset object which can be passed to a
        DataLoader.
    """

    def __init__(
        self,
        dataset: str,
        X: Tensor,
        y: Tensor,
        length: Tensor,
        split: str,
        train_prop: float,
        val_prop: float = None,
        missing: Union[float, List[float]] = 0,
        impute: Union[str, Callable[[Tensor], Tensor]] = "none",
        categorical: List[int] = [],
        override: Dict[int, float] = {},
        time: bool = True,
        mask: bool = False,
        delta: bool = False,
        downscale: float = 1.0,
        path: str = ".",
        seed: int = None,
    ) -> None:
        assert downscale >= (1.0 - EPS), "argument 'downscale' is not supported"
        assert (
            X.size(0) == y.size(0) == length.size(0)
        ), "arguments 'X', 'y' and 'length' must be the same size in dimension 0 (currently {}/{}/{} respectively)".format(  # noqa: E501
            X.size(0), y.size(0), length.size(0)
        )
        self.X = X
        self.y = y
        self.length = length
        super(TensorTimeSeriesDataset, self).__init__(
            dataset=dataset,
            split=split,
            train_prop=train_prop,
            val_prop=val_prop,
            missing=missing,
            impute=impute,
            categorical=categorical,
            override=override,
            time=time,
            mask=mask,
            delta=delta,
            downscale=downscale,
            path=path,
            seed=seed,
        )

    def _get_data(self):
        return self.X, self.y, self.length


class PhysioNet2012(_TimeSeriesDataset):
    r"""**Returns the PhysioNet Challenge 2012 data as a PyTorch Dataset.**

    See the PhysioNet `website <https://physionet.org/content/challenge-2012/1.0.0/>`_
    for a description of the data set.

    The proportion of data in the training, validation and (optional) test data sets are
    specified by the ``train_prop`` and ``val_prop`` arguments. For a
    training/validation split specify ``train_prop`` only. For a
    training/validation/test split specify both ``train_prop`` and ``val_prop``. For
    example ``train_prop=0.8`` generates a 80/20% train/validation split, but
    ``train_prop=0.8``, ``val_prop=0.1`` generates a 80/10/10% train/validation/test
    split. Splits are formed using stratified sampling.

    The ``split`` argument determines which data set is returned.

    Missing data are imputed using the ``impute`` argument. *mean* imputation replaces
    missing values with the training data channel mean. *forward* imputation replaces
    missing values with the previous observation. Alternatively a custom imputation
    function can be passed to ``impute``. This must accept ``X`` (raw time series),
    ``y`` (labels) and ``fill`` (channel means/modes) tensors and return ``X`` and ``y``
    tensors post imputation.

    The ``time``, ``mask`` and ``delta`` arguments append additional channels. By
    default, a time stamp is added as the first channel. ``mask`` adds a missing data
    mask and ``delta`` adds the time since the previous observation for each channel.
    Time deltas are calculated as in `Che et al (2018)
    <https://doi.org/10.1038/s41598-018-24271-9>`_.

    Processed data are cached in the ``./.torchtime/physionet2012`` directory by
    default. The location can be changed with the ``path`` argument, for example to
    share a single cache location across projects.

    .. note::
        When passed to a PyTorch DataLoader, batches are a named dictionary with ``X``,
        ``y`` and ``length`` data.

    Data channels are in the following order:

    :0. Mins: Minutes since ICU admission. Derived from the PhysioNet time stamp.
    :1. Albumin: Albumin (g/dL)
    :2. ALP: Alkaline phosphatase (IU/L)
    :3. ALT: Alanine transaminase (IU/L)
    :4. AST: Aspartate transaminase (IU/L)
    :5. Bilirubin: Bilirubin (mg/dL)
    :6. BUN: Blood urea nitrogen (mg/dL)
    :7. Cholesterol: Cholesterol (mg/dL)
    :8. Creatinine: Serum creatinine (mg/dL)
    :9. DiasABP: Invasive diastolic arterial blood pressure (mmHg)
    :10. FiO2: Fractional inspired O\ :sub:`2` (0-1)
    :11. GCS: Glasgow Coma Score (3-15)
    :12. Glucose: Serum glucose (mg/dL)
    :13. HCO3: Serum bicarbonate (mmol/L)
    :14. HCT: Hematocrit (%)
    :15. HR: Heart rate (bpm)
    :16. K: Serum potassium (mEq/L)
    :17. Lactate: Lactate (mmol/L)
    :18. Mg: Serum magnesium (mmol/L)
    :19. MAP: Invasive mean arterial blood pressure (mmHg)
    :20. MechVent: Mechanical ventilation respiration (0:false, or 1:true)
    :21. Na: Serum sodium (mEq/L)
    :22. NIDiasABP: Non-invasive diastolic arterial blood pressure (mmHg)
    :23. NIMAP: Non-invasive mean arterial blood pressure (mmHg)
    :24. NISysABP: Non-invasive systolic arterial blood pressure (mmHg)
    :25. PaCO2: Partial pressure of arterial CO\ :sub:`2` (mmHg)]
    :26. PaO2: Partial pressure of arterial O\ :sub:`2` (mmHg)
    :27. pH: Arterial pH (0-14)
    :28. Platelets: Platelets (cells/nL)
    :29. RespRate: Respiration rate (bpm)
    :30. SaO2: O\ :sub:`2` saturation in hemoglobin (%)
    :31. SysABP: Invasive systolic arterial blood pressure (mmHg)
    :32. Temp: Temperature (°C)
    :33. TroponinI: Troponin-I (μg/L). Note this is labelled *TropI* in the PhysioNet
        data dictionary.
    :34. TroponinT: Troponin-T (μg/L). Note this is labelled *TropT* in the PhysioNet
        data dictionary.
    :35. Urine: Urine output (mL)
    :36. WBC: White blood cell count (cells/nL)
    :37. Weight: Weight (kg)
    :38. Age: Age (years) at ICU admission
    :39. Gender: Gender (0: female, or 1: male)
    :40. Height: Height (cm) at ICU admission
    :41. ICUType: Type of ICU unit (1: Coronary Care Unit,
        2: Cardiac Surgery Recovery Unit, 3: Medical ICU, or 4: Surgical ICU)

    .. note::
        Channels 38 to 41 do not vary with time.

        Variables 11 (GCS) and 27 (pH) are assumed to be ordinal and are imputed using
        the same method as a continuous variable.

        Variable 20 (MechVent) has value ``Nan`` (the majority of values) or 1. It is
        assumed that value 1 indicates that mechanical ventilation has been used and
        ``NaN`` indicates either missing data or no mechanical ventilation. Accordingly,
        the channel mode is assumed to be zero.

    Args:
        split: The data split to return, either *train*, *val* (validation) or *test*.
        train_prop: Proportion of data in the training set.
        val_prop: Proportion of data in the validation set (optional, see above).
        impute: Method used to impute missing data, either *none*, *mean*, *forward* or
            a custom imputer function (default "none").
        time: Append time stamp in the first channel (default True).
        mask: Append missing data mask channels (default False).
        delta: Append time delta channels (default False).
        downscale: The proportion of data to return. Use to reduce the size of the data
            set when testing a model (default 1).
        path: Location of the ``.torchtime`` cache directory (default ".").
        seed: Random seed for reproducibility (optional).

    Attributes:
        X (Tensor): A tensor of default shape (*n*, *s*, *c* + 1) where *n* = number of
            trajectories, *s* = (maximum) trajectory length and *c* = number of channels
            in the PhysioNet data (*including* the time since admission in minutes). See
            above for the order of the PhysioNet channels. By default, a time stamp is
            appended as the first channel. If ``time`` is False, the time stamp is
            omitted and the tensor has shape (*n*, *s*, *c*).

            A missing data mask and/or time delta channels can be appended with the
            ``mask`` and ``delta`` arguments. These each have the same number of
            channels as the Physionet data. For example, if ``time``, ``mask`` and
            ``delta`` are all True, ``X`` has shape (*n*, *s*, 3 * *c* + 1 = 127) and
            the channels are in the order: time stamp, time series, missing data mask,
            time deltas.

            Note that PhysioNet trajectories are of unequal length and are therefore
            padded with ``NaNs`` to the length of the longest trajectory in the data.
        y (Tensor): In-hospital survival (the ``In-hospital_death`` variable) for each
            patient. *y* = 1 indicates an in-hospital death. A tensor of shape (*n*, 1).
        length (Tensor): Length of each trajectory prior to padding. A tensor of shape
            (*n*).

    .. note::
        ``X``, ``y`` and ``length`` are available for the training, validation and test
        splits by appending ``_train``, ``_val`` and ``_test`` respectively. For
        example, ``y_val`` returns the labels for the validation data set. These
        attributes are available regardless of the ``split`` argument.

    Returns:
        torch.utils.data.Dataset: A PyTorch Dataset object which can be passed to a
        DataLoader.
    """

    def __init__(
        self,
        split: str,
        train_prop: float,
        val_prop: float = None,
        impute: Union[str, Callable[[Tensor], Tensor]] = "none",
        time: bool = True,
        mask: bool = False,
        delta: bool = False,
        downscale: float = 1.0,
        path: str = ".",
        seed: int = None,
    ) -> None:
        print_message(
            "PhysioNet2012(): https://physionet.org/files/challenge-2012/1.0.0/",
            type="info",
        )
        self.DATASETS = {
            "set-a": "https://physionet.org/files/challenge-2012/1.0.0/set-a.zip?download",  # noqa: E501
            "set-b": "https://physionet.org/files/challenge-2012/1.0.0/set-b.zip?download",  # noqa: E501
            "set-c": "https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download",  # noqa: E501
        }
        self.OUTCOMES = [
            "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt?download",
            "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt?download",
            "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt?download",
        ]
        self.COLUMNS = [
            "Mins",
            "Albumin",
            "ALP",
            "ALT",
            "AST",
            "Bilirubin",
            "BUN",
            "Cholesterol",
            "Creatinine",
            "DiasABP",
            "FiO2",
            "GCS",
            "Glucose",
            "HCO3",
            "HCT",
            "HR",
            "K",
            "Lactate",
            "Mg",
            "MAP",
            "MechVent",
            "Na",
            "NIDiasABP",
            "NIMAP",
            "NISysABP",
            "PaCO2",
            "PaO2",
            "pH",
            "Platelets",
            "RespRate",
            "SaO2",
            "SysABP",
            "Temp",
            "TroponinI",
            "TroponinT",
            "Urine",
            "WBC",
            "Weight",
            "Age",
            "Gender",
            "Height",
            "ICUType",
        ]
        super(PhysioNet2012, self).__init__(
            dataset="physionet2012",
            split=split,
            train_prop=train_prop,
            val_prop=val_prop,
            impute=impute,
            categorical=[20],
            override={20: 0.0},
            time=time,
            mask=mask,
            delta=delta,
            downscale=downscale,
            path=path,
            seed=seed,
        )

    def _get_data(self):
        """Download data and form X, y, length tensors."""
        outcome_path = self.PATH / "outcomes"
        all_X = [None for _ in self.DATASETS]
        # Download and extract data
        physionet_download(self.DATASETS, self.PATH)
        [download_file(url, outcome_path) for url in self.OUTCOMES]
        # Prepare time series data
        print_message("Processing data...")
        data_directories = [self.PATH / dataset for dataset in self.DATASETS]
        data_files = get_file_list(data_directories, self.downscale, self.seed)
        length = self._get_lengths(data_directories, data_files)
        for i, files in enumerate(data_files):
            all_X[i] = self._process_files(
                data_directories[i], data_files[i], max(length), self.COLUMNS
            )
        # Prepare labels
        outcome_files = [outcome_path / file for file in os.listdir(outcome_path)]
        outcome_files.sort()
        all_y = self._get_labels(outcome_files, data_files)
        # Form tensors
        X = torch.cat(all_X)
        X[X == -1] = float("nan")  # replace -1 missing data indicator with NaNs
        y = torch.cat(all_y)
        length = torch.tensor(length)
        return X, y, length

    @staticmethod
    def _get_lengths(data_directories, data_files):
        """Get length of each time series."""
        lengths = []
        for i, files in enumerate(data_files):
            for file_j in files:
                with open(data_directories[i] / file_j) as file:
                    reader = csv.reader(file, delimiter=",")
                    lengths_j = []
                    for k, row in enumerate(reader):
                        if k > 0 and row[1] != "":  # ignore head and rows without data
                            lengths_j.append(row[0])
                    lengths_j = set(lengths_j)
                    lengths.append(len(lengths_j))
        return lengths

    @staticmethod
    def _process_files(directory, files, max_length, channels):
        """Process .txt files."""
        X = np.full((len(files), max_length, len(channels)), float("nan"))
        template_dataframe = pd.DataFrame(columns=channels)
        for i, file_i in enumerate(files):
            with open(directory / file_i) as file:
                Xi = pd.read_csv(file)
                Xi = Xi.pivot_table(index="Time", columns="Parameter", values="Value")
                Xi["Mins"] = [int(t[:2]) * 60 + int(t[3:]) for t in Xi.index]
                Xi = pd.concat([template_dataframe, Xi])
                Xi = Xi.apply(pd.to_numeric, downcast="float")
                # Add static variables
                Xi["Age"] = Xi.loc["00:00", "Age"]
                Xi["Gender"] = Xi.loc["00:00", "Gender"]
                Xi["Height"] = Xi.loc["00:00", "Height"]
                Xi["ICUType"] = Xi.loc["00:00", "ICUType"]
                X[i, : Xi.shape[0], :] = Xi[channels]
                # TODO: only include time 0 if a weight is provided
        return torch.tensor(X)

    @staticmethod
    def _get_labels(outcome_files, data_files):
        """Process outcome files."""
        y = []
        for i, file_i in enumerate(outcome_files):
            ids = data_files[i]
            with open(file_i) as file:
                y_i = pd.read_csv(
                    file, index_col=0, usecols=["RecordID", "In-hospital_death"]
                )
                ids_i = [int(id[:-4]) for id in ids]
                y_i = torch.tensor(y_i.loc[ids_i].values)
                y.append(y_i)
        return y


class PhysioNet2019(_TimeSeriesDataset):
    """**Returns the PhysioNet Challenge 2019 data as a PyTorch Dataset.**

    See the PhysioNet `website <https://physionet.org/content/challenge-2019/1.0.0/>`_
    for a description of the data set.

    The proportion of data in the training, validation and (optional) test data sets are
    specified by the ``train_prop`` and ``val_prop`` arguments. For a
    training/validation split specify ``train_prop`` only. For a
    training/validation/test split specify both ``train_prop`` and ``val_prop``. For
    example ``train_prop=0.8`` generates a 80/20% train/validation split, but
    ``train_prop=0.8``, ``val_prop=0.1`` generates a 80/10/10% train/validation/test
    split. Splits are formed using stratified sampling.

    The ``split`` argument determines which data set is returned.

    Missing data are imputed using the ``impute`` argument. *mean* imputation replaces
    missing values with the training data channel mean. *forward* imputation replaces
    missing values with the previous observation. Alternatively a custom imputation
    function can be passed to ``impute``. This must accept ``X`` (raw time series),
    ``y`` (labels) and ``fill`` (channel means/modes) tensors and return ``X`` and ``y``
    tensors post imputation.

    The ``time``, ``mask`` and ``delta`` arguments append additional channels. By
    default, a time stamp is added as the first channel. ``mask`` adds a missing data
    mask and ``delta`` adds the time since the previous observation for each channel.
    Time deltas are calculated as in `Che et al (2018)
    <https://doi.org/10.1038/s41598-018-24271-9>`_.

    Processed data are cached in the ``./.torchtime/physionet2019`` directory by
    default. The location can be changed with the ``path`` argument, for example to
    share a single cache location across projects.

    .. note::
        When passed to a PyTorch DataLoader, batches are a named dictionary with ``X``,
        ``y`` and ``length`` data.

    Args:
        split: The data split to return, either *train*, *val* (validation) or *test*.
        train_prop: Proportion of data in the training set.
        val_prop: Proportion of data in the validation set (optional, see above).
        impute: Method used to impute missing data, either *none*, *mean*, *forward* or
            a custom imputer function (default "none").
        time: Append time stamp in the first channel (default True).
        mask: Append missing data mask channels (default False).
        delta: Append time delta channels (default False).
        downscale: The proportion of data to return. Use to reduce the size of the data
            set when testing a model (default 1).
        path: Location of the ``.torchtime`` cache directory (default ".").
        seed: Random seed for reproducibility (optional).

    Attributes:
        X (Tensor): A tensor of default shape (*n*, *s*, *c* + 1) where *n* = number of
            trajectories, *s* = (maximum) trajectory length and *c* = number of channels
            in the PhysioNet data (*including* the ``ICULOS`` time stamp). The channels
            are ordered as set out on the PhysioNet `website
            <https://physionet.org/content/challenge-2019/1.0.0/>`_. By default, a
            time stamp is appended as the first channel. If ``time`` is False, the
            time stamp is omitted and the tensor has shape (*n*, *s*, *c*).

            A missing data mask and/or time delta channels can be appended with the
            ``mask`` and ``delta`` arguments. These each have the same number of
            channels as the Physionet data. For example, if ``time``, ``mask`` and
            ``delta`` are all True, ``X`` has shape (*n*, *s*, 3 * *c* + 1 = 121) and
            the channels are in the order: time stamp, time series, missing data mask,
            time deltas.

            Note that PhysioNet trajectories are of unequal length and are therefore
            padded with ``NaNs`` to the length of the longest trajectory in the data.
        y (Tensor): ``SepsisLabel`` at each time point. A tensor of shape (*n*, *s*, 1).
        length (Tensor): Length of each trajectory prior to padding. A tensor of shape
            (*n*).

    .. note::
        ``X``, ``y`` and ``length`` are available for the training, validation and test
        splits by appending ``_train``, ``_val`` and ``_test`` respectively. For
        example, ``y_val`` returns the labels for the validation data set. These
        attributes are available regardless of the ``split`` argument.

    Returns:
        torch.utils.data.Dataset: A PyTorch Dataset object which can be passed to a
        DataLoader.
    """

    def __init__(
        self,
        split: str,
        train_prop: float,
        val_prop: float = None,
        impute: Union[str, Callable[[Tensor], Tensor]] = "none",
        time: bool = True,
        mask: bool = False,
        delta: bool = False,
        downscale: float = 1.0,
        path: str = ".",
        seed: int = None,
    ) -> None:
        print_message(
            "PhysioNet2019(): https://physionet.org/files/challenge-2019/1.0.0/",
            type="info",
        )
        self.DATASETS = {
            "training": "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip",  # noqa: E501
            "training_setB": "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip",  # noqa: E501
        }
        super(PhysioNet2019, self).__init__(
            dataset="physionet2019",
            split=split,
            train_prop=train_prop,
            val_prop=val_prop,
            impute=impute,
            time=time,
            mask=mask,
            delta=delta,
            downscale=downscale,
            path=path,
            seed=seed,
        )

    def _get_data(self):
        """Download data and form X, y, length tensors."""
        # Download and extract data
        physionet_download(self.DATASETS, self.PATH)
        # Prepare data
        print_message("Processing data...")
        data_directories = [self.PATH / dataset for dataset in self.DATASETS]
        data_files = get_file_list(data_directories, self.downscale, self.seed)
        length, channels = self._get_lengths_channels(data_directories, data_files)
        all_X = [None for _ in self.DATASETS]
        all_y = [None for _ in self.DATASETS]
        for i, files in enumerate(data_files):
            all_X[i], all_y[i] = self._process_files(
                data_directories[i], files, max(length), channels
            )
        # Form tensors
        X = torch.cat(all_X)
        y = torch.cat(all_y)
        length = torch.tensor(length)
        return X, y, length

    @staticmethod
    def _get_lengths_channels(data_directories, data_files, max_time=None):
        """Get length of each time series and number of channels. Time series can be
        truncated at a specific hour with the ``max_time`` argument."""
        lengths = []  # sequence lengths
        channels = []  # number of channels
        for i, files in enumerate(data_files):
            for file_j in files:
                with open(data_directories[i] / file_j) as file:
                    reader = csv.reader(file, delimiter="|")
                    lengths_j = []
                    for k, Xijk in enumerate(reader):
                        channels.append(len(Xijk))
                        if k > 0:  # ignore header
                            if max_time:
                                if int(Xijk[39]) <= max_time:
                                    lengths_j.append(1)
                            else:
                                lengths_j.append(1)
                    lengths.append(sum(lengths_j))
        channels = list(set(channels))
        assert len(channels) == 1, "corrupt file, delete data and re-run"
        return lengths, channels[0]

    @staticmethod
    def _process_files(directory, files, max_length, channels):
        """Process .psv files."""
        X = np.full((len(files), max_length, channels - 1), float("nan"))
        y = np.full((len(files), max_length, 1), float("nan"))
        for i, file_i in enumerate(files):
            with open(directory / file_i) as file:
                reader = csv.reader(file, delimiter="|")
                for j, Xij in enumerate(reader):
                    if j > 0:  # ignore header
                        X[i, j - 1] = Xij[:-1]
                        y[i, j - 1, 0] = Xij[-1]
        return torch.tensor(X), torch.tensor(y)


class PhysioNet2019Binary(_TimeSeriesDataset):
    """**Returns simplified binary prediction version of the PhysioNet Challenge 2019
    data as a PyTorch Dataset.**

    In contrast with the full challenge, the first 72 hours of data are used to predict
    whether a patient develops sepsis at any point during the period of hospitalisation
    as in `Kidger et al (2020) <https://arxiv.org/abs/2005.08926>`_.

    See the PhysioNet `website <https://physionet.org/content/challenge-2019/1.0.0/>`_
    for a description of the data set.

    The proportion of data in the training, validation and (optional) test data sets are
    specified by the ``train_prop`` and ``val_prop`` arguments. For a
    training/validation split specify ``train_prop`` only. For a
    training/validation/test split specify both ``train_prop`` and ``val_prop``. For
    example ``train_prop=0.8`` generates a 80/20% train/validation split, but
    ``train_prop=0.8``, ``val_prop=0.1`` generates a 80/10/10% train/validation/test
    split. Splits are formed using stratified sampling.

    The ``split`` argument determines which data set is returned.

    Missing data are imputed using the ``impute`` argument. *mean* imputation replaces
    missing values with the training data channel mean. *forward* imputation replaces
    missing values with the previous observation. Alternatively a custom imputation
    function can be passed to ``impute``. This must accept ``X`` (raw time series),
    ``y`` (labels) and ``fill`` (channel means/modes) tensors and return ``X`` and ``y``
    tensors post imputation.

    The ``time``, ``mask`` and ``delta`` arguments append additional channels. By
    default, a time stamp is added as the first channel. ``mask`` adds a missing data
    mask and ``delta`` adds the time since the previous observation for each channel.
    Time deltas are calculated as in `Che et al (2018)
    <https://doi.org/10.1038/s41598-018-24271-9>`_.

    Processed data are cached in the ``./.torchtime/physionet2019`` directory by
    default. The location can be changed with the ``path`` argument, for example to
    share a single cache location across projects.

    .. note::
        When passed to a PyTorch DataLoader, batches are a named dictionary with ``X``,
        ``y`` and ``length`` data.

    Args:
        split: The data split to return, either *train*, *val* (validation) or *test*.
        train_prop: Proportion of data in the training set.
        val_prop: Proportion of data in the validation set (optional, see above).
        impute: Method used to impute missing data, either *none*, *mean*, *forward* or
            a custom imputer function (default "none").
        time: Append time stamp in the first channel (default True).
        mask: Append missing data mask channels (default False).
        delta: Append time delta channels (default False).
        downscale: The proportion of data to return. Use to reduce the size of the data
            set when testing a model (default 1).
        path: Location of the ``.torchtime`` cache directory (default ".").
        seed: Random seed for reproducibility (optional).

    Attributes:
        X (Tensor): A tensor of default shape (*n*, *s*, *c* + 1) where *n* = number of
            trajectories, *s* = (maximum) trajectory length and *c* = number of channels
            in the PhysioNet data (*including* the ``ICULOS`` time stamp). The channels
            are ordered as set out on the PhysioNet `website
            <https://physionet.org/content/challenge-2019/1.0.0/>`_. By default, a
            time stamp is appended as the first channel. If ``time`` is False, the
            time stamp is omitted and the tensor has shape (*n*, *s*, *c*).

            A missing data mask and/or time delta channels can be appended with the
            ``mask`` and ``delta`` arguments. These each have the same number of
            channels as the Physionet data. For example, if ``time``, ``mask`` and
            ``delta`` are all True, ``X`` has shape (*n*, *s*, 3 * *c* + 1 = 121) and
            the channels are in the order: time stamp, time series, missing data mask,
            time deltas.

            Note that PhysioNet trajectories are of unequal length and are therefore
            padded with ``NaNs`` to the length of the longest trajectory in the data.
        y (Tensor): Whether patient is diagnosed with sepsis at any time during
            hospitalisation. A tensor of shape (*n*, 1).
        length (Tensor): Length of each trajectory prior to padding. A tensor of shape
            (*n*).

    .. note::
        ``X``, ``y`` and ``length`` are available for the training, validation and test
        splits by appending ``_train``, ``_val`` and ``_test`` respectively. For
        example, ``y_val`` returns the labels for the validation data set. These
        attributes are available regardless of the ``split`` argument.

    Returns:
        torch.utils.data.Dataset: A PyTorch Dataset object which can be passed to a
        DataLoader.
    """

    def __init__(
        self,
        split: str,
        train_prop: float,
        val_prop: float = None,
        impute: Union[str, Callable[[Tensor], Tensor]] = "none",
        time: bool = True,
        mask: bool = False,
        delta: bool = False,
        downscale: float = 1.0,
        path: str = ".",
        seed: int = None,
    ) -> None:
        print_message(
            "PhysioNet2019Binary(): https://physionet.org/files/challenge-2019/1.0.0/",
            type="info",
        )
        self.DATASETS = {
            "training": "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip",  # noqa: E501
            "training_setB": "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip",  # noqa: E501
        }
        self.path = path
        self.max_time = 72  # hours
        super(PhysioNet2019Binary, self).__init__(
            dataset="physionet2019binary",
            split=split,
            train_prop=train_prop,
            val_prop=val_prop,
            impute=impute,
            time=time,
            mask=mask,
            delta=delta,
            downscale=downscale,
            path=path,
            seed=seed,
        )

    def _get_data(self):
        """Download data and form X, y, length tensors."""
        # Download and extract data in "physionet2019" directory to avoid duplication
        self.PATH = pathlib.Path() / self.path / ".torchtime" / "physionet2019"
        physionet_download(self.DATASETS, self.PATH)
        # Prepare data
        print_message("Processing data...")
        data_directories = [self.PATH / dataset for dataset in self.DATASETS]
        data_files = get_file_list(data_directories, self.downscale, self.seed)
        length, channels = PhysioNet2019._get_lengths_channels(
            data_directories, data_files, max_time=self.max_time
        )

        print([i for i, x in enumerate(length) if x == 0])

        all_X = [None for _ in self.DATASETS]
        all_y = [None for _ in self.DATASETS]
        for i, files in enumerate(data_files):
            all_X[i], all_y[i] = self._process_files(
                data_directories[i], files, max(length), channels
            )
        # Form tensors
        X = torch.cat(all_X)
        y = torch.cat(all_y)
        length = torch.tensor(length)
        # Drop patients with zero length sequences
        patient_index = torch.arange(X.size(0)).masked_select(length != 0).int()
        X = X.index_select(index=patient_index, dim=0)
        y = y.index_select(index=patient_index, dim=0)
        length = length.index_select(index=patient_index, dim=0)
        # Save cached files to "physionet2019binary" directory
        self.PATH = pathlib.Path() / self.path / ".torchtime" / "physionet2019binary"
        return X, y, length

    def _process_files(self, directory, files, max_length, channels):
        """Process .psv files."""
        X = np.full((len(files), max_length, channels - 1), float("nan"))
        y = np.full((len(files), 1), 0.0)
        for i, file_i in enumerate(files):
            with open(directory / file_i) as file:
                reader = csv.reader(file, delimiter="|")
                for j, Xij in enumerate(reader):
                    if j > 0:  # ignore header
                        if int(Xij[39]) <= self.max_time:
                            X[i, j - 1] = Xij[:-1]
                        y[i, 0] = max(y[i, 0], int(Xij[-1]))  # sepsis at any point
        return torch.tensor(X), torch.tensor(y)


class UEA(_TimeSeriesDataset):
    """
    **Returns a time series classification data set from the UEA/UCR
    repository as a PyTorch Dataset.**

    See the UEA/UCR repository `website
    <https://www.timeseriesclassification.com/>`_ for a list of data sets and
    their descriptions.

    The proportion of data in the training, validation and (optional) test data sets are
    specified by the ``train_prop`` and ``val_prop`` arguments. For a
    training/validation split specify ``train_prop`` only. For a
    training/validation/test split specify both ``train_prop`` and ``val_prop``. For
    example ``train_prop=0.8`` generates a 80/20% train/validation split, but
    ``train_prop=0.8``, ``val_prop=0.1`` generates a 80/10/10% train/validation/test
    split. Splits are formed using stratified sampling.

    The ``split`` argument determines which data set is returned.

    Missing data can be simulated using the ``missing`` argument. Data are dropped at
    random.

    Missing data are imputed using the ``impute`` argument. *mean* imputation replaces
    missing values with the training data channel mean. *forward* imputation replaces
    missing values with the previous observation. Alternatively a custom imputation
    function can be passed to ``impute``. This must accept ``X`` (raw time series),
    ``y`` (labels) and ``fill`` (channel means/modes) tensors and return ``X`` and ``y``
    tensors post imputation.

    .. warning::
        Mean and forward imputation are unsuitable for categorical variables. To impute
        missing values for a categorical variable with the channel mode (rather than
        the channel mean), pass the channel indices to the ``categorical`` argument.
        Note this is also required for forward imputation to appropriately impute
        initial missing values.

        Alternatively, the calculated channel mean/mode can be overridden using the
        ``override`` argument. This can be used to impute missing data with a fixed
        value.

    The ``time``, ``mask`` and ``delta`` arguments append additional channels. By
    default, a time stamp is added as the first channel. ``mask`` adds a missing data
    mask and ``delta`` adds the time since the previous observation for each channel.
    Time deltas are calculated as in `Che et al (2018)
    <https://doi.org/10.1038/s41598-018-24271-9>`_.

    Data are downloaded using the ``sktime`` package and cached in the
    ``./.torchtime/[dataset name]`` directory by default. The location can be changed
    with the ``path`` argument, for example to share a single cache location across
    projects.

    .. note::
        When passed to a PyTorch DataLoader, batches are a named dictionary with ``X``,
        ``y`` and ``length`` data.

    Args:
        dataset: The data set to return.
        split: The data split to return, either *train*, *val* (validation) or *test*.
        train_prop: Proportion of data in the training set.
        val_prop: Proportion of data in the validation set (optional, see above).
        missing: The proportion of data to drop at random. If ``missing`` is a single
            value, data are dropped from all channels. To drop data independently across
            each channel, pass a list of rates with the proportion missing for each
            channel (default 0 i.e. no missing data simulation).
        impute: Method used to impute missing data, either *none*, *mean*, *forward* or
            a custom imputer function (default "none").
        categorical: List with channel indices of categorical variables (default ``[]``
            i.e. no categorical variables).
        override: Override the calculated channel mean/mode when imputing data.
            Dictionary with channel indices and values e.g. {1: 4.5, 3: 7.2} (default
            ``{}`` i.e. no overridden channel mean/modes).
        time: Append time stamp in the first channel (default True).
        mask: Append missing data mask channels (default False).
        delta: Append time delta channels (default False).
        downscale: The proportion of data to return. Use to reduce the size of the data
            set when testing a model (default 1).
        path: Location of the ``.torchtime`` cache directory (default ".").
        seed: Random seed for reproducibility (optional).

    Attributes:
        X (Tensor): A tensor of default shape (*n*, *s*, *c* + 1) where *n* = number of
            trajectories, *s* = (maximum) trajectory length and *c* = number of
            channels. By default, a time stamp is appended as the first channel. If
            ``time`` is False, the time stamp is omitted and the tensor has shape
            (*n*, *s*, *c*).

            A missing data mask and/or time delta channels can be appended with the
            ``mask`` and ``delta`` arguments. These each have the same number of
            channels as the data set. For example, if ``time``, ``mask`` and
            ``delta`` are all True, ``X`` has shape (*n*, *s*, 3 * *c* + 1) and the
            channels are in the order: time stamp, time series, missing data mask, time
            deltas.

            Where trajectories are of unequal lengths they are padded with ``NaNs`` to
            the length of the longest trajectory in the data.
        y (Tensor): One-hot encoded label data. A tensor of shape (*n*, *l*) where *l*
            is the number of classes.
        length (Tensor): Length of each trajectory prior to padding. A tensor of shape
            (*n*).

    .. note::
        ``X``, ``y`` and ``length`` are available for the training, validation and test
        splits by appending ``_train``, ``_val`` and ``_test`` respectively. For
        example, ``y_val`` returns the labels for the validation data set. These
        attributes are available regardless of the ``split`` argument.

    Returns:
        torch.utils.data.Dataset: A PyTorch Dataset object which can be passed to a
        DataLoader.
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        train_prop: float,
        val_prop: float = None,
        missing: Union[float, List[float]] = 0,
        impute: Union[str, Callable[[Tensor], Tensor]] = "none",
        categorical: List[int] = [],
        override: Dict[int, float] = {},
        time: bool = True,
        mask: bool = False,
        delta: bool = False,
        downscale: float = 1.0,
        path: str = ".",
        seed: int = None,
    ) -> None:
        print_message(
            "UEA(dataset="
            + dataset
            + "): https://www.timeseriesclassification.com/description.php?Dataset="
            + dataset,
            type="info",
        )
        self.dataset = dataset
        self.max_length = 0
        super(UEA, self).__init__(
            dataset=dataset,
            split=split,
            train_prop=train_prop,
            val_prop=val_prop,
            missing=missing,
            impute=impute,
            categorical=categorical,
            override=override,
            time=time,
            mask=mask,
            delta=delta,
            downscale=downscale,
            path=path,
            seed=seed,
        )

    def _get_data(self):
        """Download data and form ``X``, ``y`` and ``length`` tensors."""
        X_raw, y_raw = load_UCR_UEA_dataset(self.dataset)
        print_message("Processing data...")
        if self.downscale < (1.0 - EPS):
            idx = sample_indices(len(X_raw), self.downscale, seed=self.seed)
            X_raw = X_raw.iloc[idx]
            y_raw = y_raw[idx]
        # Length of each trajectory
        channel_lengths = X_raw.apply(lambda Xi: Xi.apply(len), axis=1)
        length = torch.tensor(channel_lengths.apply(max, axis=1).values)
        self.max_length = length.max()
        # Form tensor with padded trajectories
        X = torch.stack(
            [self._pad(X_raw.iloc[i]) for i in range(len(X_raw))],
            dim=0,
        )
        # One-hot encode labels (start from zero)
        y = torch.tensor(y_raw.astype(int))
        if all(y != 0):
            y -= 1
        y = F.one_hot(y)
        return X, y, length

    def _pad(self, Xi):
        """Pad trajectories to maximum length in data."""
        Xi = pad_sequence([torch.tensor(Xij) for Xij in Xi])
        out = torch.full((self.max_length, Xi.size(1)), float("nan"))  # shape (s, c)
        out[0 : Xi.size(0)] = Xi
        return out
