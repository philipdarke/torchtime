import csv
import os
import pathlib
import tempfile
import zipfile
from typing import Callable, List, Union

import numpy as np
import requests
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sktime.datasets import load_UCR_UEA_dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from torchtime.impute import forward_impute, replace_missing


class _TimeSeriesDataset(Dataset):
    """**Generic time series PyTorch Dataset.**

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
    missing values with the channel mean in the training data. *forward* imputation
    replaces missing values with the previous observation. Alternatively a custom
    imputation function can be passed to ``impute``. This must accept ``X`` and ``y``
    tensors with the raw time series and labels respectively and return both tensors
    post imputation.

    The ``time``, ``mask`` and ``delta`` arguments append additional channels. By
    default, a time stamp is added as the first channel. ``mask`` adds a missing data
    mask and ``delta`` adds the time since the previous observation for each channel.

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
            a custom imputer function (default *none*).
        time: Append time stamp in the first channel (default True).
        mask: Append missing data mask channels (default False).
        delta: Append time delta channels calculated as in `Che et al (2018)
            <https://doi.org/10.1038/s41598-018-24271-9>`_ (default False).
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
        time: bool = True,
        mask: bool = False,
        delta: bool = False,
        downscale: float = 1.0,
        path: str = ".",
        seed: int = None,
    ) -> None:
        """
        Data processing pipeline:

            1. Download data and prepare X, y, length tensors
            2. Downsample data set for testing if ``downscale`` argument passed
            3. Simulate missing data
            4. Add time/missing data mask/time delta channels
            5. Split into training/validation/test data sets
            6. Impute missing data
            7. Assign ``split`` to X, y, length attributes
        """
        self.val_prop = val_prop
        self.missing = missing
        self.time = time
        self.mask = mask
        self.seed = seed
        self.n_channels = 0
        self.test_prop = 0

        # Constants
        self.PATH = pathlib.Path() / path / ".torchtime" / dataset
        self.EPS = np.finfo(float).eps
        self.IMPUTE_FUNCTIONS = {
            "none": self._no_imputation,
            "mean": self._mean_imputation,
            "forward": self._forward_imputation,
        }

        # Validate impute argument
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

        # Data splits
        assert (
            train_prop > self.EPS and train_prop < 1
        ), "argument 'train_prop' must be in range (0, 1)"
        if self.val_prop is None:
            self.val_prop = 1 - train_prop
        else:
            assert (
                self.val_prop > self.EPS and self.val_prop < 1 - train_prop
            ), "argument 'val_prop' must be in range (0, 1-train_prop)"
            self.test_prop = 1 - train_prop - self.val_prop
            self.val_prop = self.val_prop / (1 - self.test_prop)

        # Validate split argument
        splits = ["train", "val"]
        if self.test_prop > self.EPS:
            splits.append("test")
        assert split in splits, "argument 'split' must be one of {}".format(splits)

        # 1. Get data from cache, else call _get_data() and cache results
        if self._check_cache():
            X_all, y_all, length_all = self._load_cache()
        else:
            X_all, y_all, length_all = self._get_data()
            X_all = X_all.float()  # float32 precision
            y_all = y_all.float()  # float32 precision
            length_all = length_all.long()  # int64 precision
            self._cache_data(X_all, y_all, length_all)

        # 2. Downscale data set
        if downscale < (1 - self.EPS):
            mask_indices = self._random_mask(X_all.size(0), downscale)
            X_all = X_all[mask_indices]
            y_all = y_all[mask_indices]
            length_all = length_all[mask_indices]

        # 3. Simulate missing data
        self._simulate_missing(X_all)

        # 4. Add time stamp/mask/time delta channels
        if time:
            X_all = torch.cat([self._time_stamp(X_all), X_all], dim=2)
        if mask:
            X_all = torch.cat([X_all, self._missing_mask(X_all)], dim=2)
        if delta:
            X_all = torch.cat([X_all, self._time_delta(X_all)], dim=2)

        # 5. Form train/validation/test splits
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

        # 6. Impute missing data (no missing values in time, mask and delta channels)
        self.train_means = torch.nanmean(torch.flatten(self.X_train, end_dim=1), 0)
        self.X_train, self.y_train = imputer(self.X_train, self.y_train)
        self.X_val, self.y_val = imputer(self.X_val, self.y_val)
        if self.test_prop > self.EPS:
            self.X_test, self.y_test = imputer(self.X_test, self.y_test)
        else:
            del self.X_test, self.y_test, self.length_test

        # 7. Return data split
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

    def _random_mask(self, length, proportion, generator=None):
        """Mask to select ``proportion`` from a batch of size ``length``."""
        if generator is None:
            generator = torch.Generator()
            if self.seed is not None:
                generator = generator.manual_seed(self.seed)
        subset_size = int(length * proportion)
        return torch.randperm(length, generator=generator)[:subset_size]

    def _no_imputation(self, X, y):
        """No imputation."""
        return X, y

    def _mean_imputation(self, X, y):
        """Mean imputation. Replace missing values with channel means in ``X`` and zeros
        in ``y``."""
        X_imputed = replace_missing(X, fill=self.train_means)
        y_imputed = replace_missing(y, fill=torch.zeros(y.size(-1)))
        return X_imputed, y_imputed

    def _forward_imputation(self, X, y):
        """Forward imputation. Replace missing values with previous observation. Replace
        any initial missing values with channel means in ``X``. Assume no missing
        initial values in ``y`` but there may be trailing missing values due to
        padding."""
        X_imputed = forward_impute(X, fill=self.train_means)
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

    def _download_zip(self, url, path):
        """Download and extract a .zip file from ``url`` to ``path``."""
        print("Downloading ", url, "...", sep="")
        with tempfile.NamedTemporaryFile() as temp_file:
            download = requests.get(url)
            temp_file.write(download.content)
            zipfile.ZipFile(temp_file, "r").extractall(path)

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
                mask_indices = self._random_mask(length, self.missing, generator)
                Xi[mask_indices] = float("nan")
            else:
                assert Xi.size(-1) == len(
                    self.missing
                ), "argument 'missing' must be same length as number of channels \
                    ({})".format(
                    Xi.size(-1)
                )
                for channel, rate in enumerate(self.missing):
                    mask_indices = self._random_mask(length, rate, generator)
                    Xi[mask_indices, channel] = float("nan")

    def _time_stamp(self, X):
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
        if self.test_prop > self.EPS:
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
    **Returns a time series data set from input tensors as a PyTorch Dataset.**

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
    missing values with the channel mean in the training data. *forward* imputation
    replaces missing values with the previous observation. Alternatively a custom
    imputation function can be passed to ``impute``. This must accept ``X`` and ``y``
    tensors with the raw time series and labels respectively and return both tensors
    post imputation.

    The ``time``, ``mask`` and ``delta`` arguments append additional channels. By
    default, a time stamp is added as the first channel. ``mask`` adds a missing data
    mask and ``delta`` adds the time since the previous observation for each channel.

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
            a custom imputer function (default *none*).
        time: Append time stamp in the first channel (default True).
        mask: Append missing data mask channels (default False).
        delta: Append time delta channels calculated as in `Che et al (2018)
            <https://doi.org/10.1038/s41598-018-24271-9>`_ (default False).
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
        time: bool = True,
        mask: bool = False,
        delta: bool = False,
        downscale: float = 1.0,
        path: str = ".",
        seed: int = None,
    ) -> None:
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
            time=time,
            mask=mask,
            delta=delta,
            downscale=downscale,
            path=path,
            seed=seed,
        )

    def _get_data(self):
        return self.X, self.y, self.length


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
    missing values with the channel mean in the training data. *forward* imputation
    replaces missing values with the previous observation. Alternatively a custom
    imputation function can be passed to ``impute``. This must accept ``X`` and ``y``
    tensors with the raw time series and labels respectively and return both tensors
    post imputation.

    The ``time``, ``mask`` and ``delta`` arguments append additional channels. By
    default, a time stamp is added as the first channel. ``mask`` adds a missing data
    mask and ``delta`` adds the time since the previous observation for each channel.

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
            a custom imputer function (default *none*).
        time: Append time stamp in the first channel (default True).
        mask: Append missing data mask channels (default False).
        delta: Append time delta channels calculated as in `Che et al (2018)
            <https://doi.org/10.1038/s41598-018-24271-9>`_ (default False).
        downscale: The proportion of data to return. Use to reduce the size of the data
            set when testing a model (default 1).
        path: Location of the ``.torchtime`` cache directory (default ".").
        seed: Random seed for reproducibility (optional).

    Attributes:
        X (Tensor): A tensor of default shape (*n*, *s*, *c* + 1) where *n* = number of
            trajectories, *s* = (maximum) trajectory length and *c* = number of channels
            in the PhysioNet data (*including* the ``ICULOS`` time stamp). By default, a
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
        self.DATASETS = {
            "training": "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip",  # noqa: E501
            "training_setB": "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip",  # noqa: E501
        }
        super(PhysioNet2019, self).__init__(
            dataset="physionet2019",
            split=split,
            train_prop=train_prop,
            val_prop=val_prop,
            missing=0,
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
        all_data = [None] * len(self.DATASETS)
        for i, url in enumerate(self.DATASETS.values()):
            with tempfile.TemporaryDirectory() as temp_dir:
                self._download_zip(url, temp_dir)
                all_data[i] = self._process_set(temp_dir)
        # Form tensors
        X = torch.cat([X for X, _ in all_data])  # may fail if data is updated
        length = torch.cat([length for _, length in all_data])
        y = X[:, :, -1].unsqueeze(2)
        X = X[:, :, :-1]
        return X, y, length

    def _process_set(self, path):
        """Process ``.psv`` files."""
        # Get n = number of trajectories, s = longest trajectory, c = number of
        # channels from data
        subpath = pathlib.Path() / next(os.scandir(path)).path  # extracted directory
        n, s, c = len([file for file in os.listdir(subpath)]), [], []
        for filename in os.listdir(subpath):
            with open(subpath / filename) as file:
                reader = csv.reader(file, delimiter="|")
                s_i = []
                for j, row in enumerate(reader):
                    if j == 0:
                        c.append(len(row))
                    else:
                        s_i.append(1)
                s.append(sum(s_i))
        c = list(set(c))
        assert len(c) == 1, "corrupt file, delete data and re-run"
        c = c[0]
        # Extract data from each file
        X = np.full((n, max(s), c), float("nan"))
        for i, filename in enumerate(os.listdir(subpath)):
            with open(subpath / filename) as file:
                reader = csv.reader(file, delimiter="|")
                for j, Xij in enumerate(reader):
                    if j > 0:
                        X[i, j - 1] = Xij
        return torch.tensor(X), torch.tensor(s)


class UEA(_TimeSeriesDataset):
    """
    **Returns a time series classification data set from the UEA/UCR
    repository as a PyTorch Dataset.**

    See the UEA/UCR repository `website
    <https://physionet.org/content/challenge-2019/1.0.0/>`_ for a list of data sets and
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
    missing values with the channel mean in the training data. *forward* imputation
    replaces missing values with the previous observation. Alternatively a custom
    imputation function can be passed to ``impute``. This must accept ``X`` and ``y``
    tensors with the raw time series and labels respectively and return both tensors
    post imputation.

    The ``time``, ``mask`` and ``delta`` arguments append additional channels. By
    default, a time stamp is added as the first channel. ``mask`` adds a missing data
    mask and ``delta`` adds the time since the previous observation for each channel.

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
            a custom imputer function (default *none*).
        time: Append time stamp in the first channel (default True).
        mask: Append missing data mask channels (default False).
        delta: Append time delta channels calculated as in `Che et al (2018)
            <https://doi.org/10.1038/s41598-018-24271-9>`_ (default False).
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
        time: bool = True,
        mask: bool = False,
        delta: bool = False,
        downscale: float = 1.0,
        path: str = ".",
        seed: int = None,
    ) -> None:
        self.dataset = dataset
        self.max_length = 0
        super(UEA, self).__init__(
            dataset=dataset,
            split=split,
            train_prop=train_prop,
            val_prop=val_prop,
            missing=missing,
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
        print("Downloading ", self.dataset, "...", sep="")
        X_raw, y_raw = load_UCR_UEA_dataset(self.dataset)
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
