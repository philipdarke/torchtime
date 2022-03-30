import csv
import os
import pathlib
import tempfile
import zipfile
from typing import List, Union

import numpy as np
import requests
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sktime.datasets import load_UCR_UEA_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _missing(X, missing, seed):
    """Simulate missing data."""
    length = X.size(1)
    generator = torch.Generator()
    if seed is not None:
        generator = generator.manual_seed(seed)
    for Xi in X:
        if type(missing) in [int, float]:
            n_samples = int(length * missing)
            mask = torch.randperm(length, generator=generator)[:n_samples]
            Xi[mask] = float("nan")
        else:
            assert Xi.size(-1) == len(
                missing
            ), "argument 'missing' must be same length as number of channels \
                ({})".format(
                Xi.size(-1)
            )
            for channel, rate in enumerate(missing):
                n_samples = int(length * rate)
                mask = torch.randperm(length, generator=generator)[:n_samples]
                Xi[mask, channel] = float("nan")
    return X


def _add_time(X):
    """Add time stamp in channel 1."""
    time_stamp = torch.arange(X.size(1)).unsqueeze(0)
    time_stamp = time_stamp.tile((X.size(0), 1)).unsqueeze(2)
    return torch.cat([time_stamp, X], dim=2)


def _add_mask(X, time):
    """Add missing data mask after trajectory data."""
    mask = torch.logical_not(torch.isnan(X[:, :, time:]))
    X = torch.cat([X, mask], dim=2)
    return X


def _add_time_delta(X, time, mask):
    """Add time delta calculated as in Che et al, 2018, see
    https://www.nature.com/articles/s41598-018-24271-9. Assumes X includes a time stamp
    and a missing data mask."""
    # Add time and mask to X as required
    if not time:
        X = _add_time(X)
    if not mask:
        X = _add_mask(X, not time)
    n_channels = int((X.size(-1) - 1) / 2)
    # Calculate time delta channels
    X = X.transpose(1, 2)  # shape (n, c, s)
    time_stamp = X[:, 0].unsqueeze(1).repeat(1, n_channels, 1)
    time_delta = time_stamp.clone()
    time_mask = X[:, -n_channels:].clone()
    time_delta[:, :, 0] = 0  # first time delta is zero by definition
    time_mask[:, :, 0] = 1
    time_mask = torch.cummax(time_mask, -1)[1]
    time_delta = time_delta.gather(
        -1, time_mask
    )  # time of previous observation if data missing
    time_delta = torch.cat(
        (
            time_delta[:, :, 0].unsqueeze(2),
            time_stamp[:, :, 1:] - time_delta[:, :, :-1],
        ),
        dim=2,
    )  # convert to delta
    # Remove time/mask from X as required
    X = X[:, int(not time) : (n_channels * -~mask + 1)]
    # Add time delta and return
    X = torch.cat((X, time_delta), 1)
    return X.transpose(1, 2)  # shape (n, s, c)


def _split_data(X, y, length, stratify, split, val_split, test_split, seed):
    """Split data (X, y, length) into training, validation and (optional) test sets
    using stratified sampling."""
    if test_split > np.finfo(float).eps:
        # Test split
        test_nontest_split = train_test_split(
            X,
            y,
            length,
            stratify,
            train_size=test_split,
            random_state=seed,
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
            train_size=val_split,
            random_state=seed,
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
            train_size=val_split,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        X_val, X_train, y_val, y_train, length_val, length_train = val_train_split
        X_test, y_test, length_test = float("nan"), float("nan"), float("nan")
    # Return splits
    if split == "test":
        X = X_test
        y = y_test
        length = length_test
    elif split == "train":
        X = X_train
        y = y_train
        length = length_train
    else:
        X = X_val
        y = y_val
        length = length_val
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
        X,
        y,
        length,
    )


class UEA(Dataset):
    """
    Creates and returns a time series classification data set from the UEA/UCR
    repository as a PyTorch DataSet. See the `repository
    <http://www.timeseriesclassification.com>`_ for more details and the available data
    sets.

    Data are downloaded using the ``sktime`` package. The size of the training,
    validation and (optional) test splits are controlled using the ``train_split`` and
    ``val_split`` arguments. For a training/validation split use ``train_split`` only.
    For a training/validation/test split specify both ``train_split`` and
    ``val_split``. For example ``train_split = 0.8`` generates a 80/20%
    train/validation split and ``train_split = 0.7``, ``val_split = 0.2`` generates a
    70/20/10% train/validation/test split. Splits are formed using stratified sampling.

    Missing data can be simulated using the ``missing`` argument. Randomness is
    controlled with the ``seed`` argument.

    The ``X``, ``y`` and ``length`` attributes correspond to the split specified by the
    ``split`` argument. However training, validation and test data are available by
    accessing the ``X_[train/val/test]``, ``y_[train/val/test]`` and
    ``length__[train/val/test]`` attributes regardless of the split specified in
    ``split``.

    The ``time``, ``mask`` and ``delta`` arguments are used to append a time stamp
    channel (added by default as the first channel), missing data mask and time delta
    channels. The time delta represents the time since the previous observation
    calculated as in `Che et al (2018) <https://doi.org/10.1038/s41598-018-24271-9>`_.

    Args:
        dataset: The data set to return.
        split: The data split to return, either 'train', 'val' (validation) or 'test'.
        train_split: Proportion of data in the training set.
        val_split: Proportion of data in the validation set (optional, see above).
        missing: The proportion of data to drop at random. If ``missing`` is a single
            value, data is dropped from all channels. To drop data at different rates
            for each channel, pass a list of rates with the proportion missing for each
            channel (default None).
        time: Add time stamp in the first channel (default True).
        mask: Add missing data mask (default False).
        delta: Add time delta (default False).
        seed: Random seed for reproducibility (optional).

    Attributes:
        X (Tensor): A tensor of default shape (*n*, *s*, *c* + 1) where *n* = number of
            trajectories, *s* = (maximum) trajectory length and *c* = number of
            channels. By default, a time stamp is appended as the first channel. If
            ``time`` is False, the time stamp is omitted and the tensor has shape
            (*n*, *s*, *c*).

            A missing data mask and/or time delta channels can be added with the
            ``mask`` and ``delta`` arguments. These each have the same number of
            channels as the data set. For example, if ``time``, ``mask`` and
            ``delta`` are all True, ``X`` has shape (*n*, *s*, 3 * *c* + 1).

            Where trajectories are of unequal lengths they are padded with ``NaNs`` to
            the length of the longest trajectory in the data.
        y (Tensor): One-hot encoded label data. A tensor of shape (*n*, *l*) where *l*
            is the number of classes.
        length (Tensor): Length of each trajectory prior to padding. A tensor of shape
            (*n*, 1).

    Returns:
        torch.utils.data.Dataset: A PyTorch DataSet object which can be passed to a
        DataLoader.
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        train_split: float,
        val_split: float = None,
        missing: Union[float, List[float]] = None,
        time: bool = True,
        mask: bool = False,
        delta: bool = False,
        seed: int = None,
    ) -> None:
        # Data splits
        if val_split is None:
            test_split = 0
            val_split = 1 - train_split
        else:
            test_split = 1 - train_split - val_split
            val_split = val_split / (1 - test_split)
        # Check arguments
        splits = ["train", "val"]
        if test_split > np.finfo(float).eps:
            splits.append("test")
        assert split in splits, "argument 'split' must be one of {}".format(splits)
        if missing is not None and type(missing) not in [int, float, list]:
            raise AssertionError("argument 'missing' must be a int/float or a list")
        # Get data and form tensors
        self.X_raw, self.y_raw = load_UCR_UEA_dataset(dataset)
        traj_lengths = self.X_raw.apply(
            lambda Xi: Xi.apply(len), axis=1  # length of each channel
        )
        self.length = torch.tensor(
            traj_lengths.apply(max, axis=1).values  # length of each trajectory
        )
        max_length = self.length.max()
        self.X = torch.stack(
            [self._pad(self.X_raw.iloc[i], max_length) for i in range(len(self.X_raw))],
            dim=0,
        )
        self.y = torch.tensor(self.y_raw.astype(int))
        # One-hot encode labels
        if all(self.y != 0):
            self.y -= 1  # start class labels from zero
        self.y = F.one_hot(self.y)
        # Simulate missing data
        if missing is not None:
            self.X = _missing(self.X, missing, seed)
        # Add time stamp/mask/time delta channels
        if time:
            self.X = _add_time(self.X)
        if mask:
            self.X = _add_mask(self.X, time)
        if delta:
            self.X = _add_time_delta(self.X, time, mask)
        # Form test/train/validation splits
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
            self.X,
            self.y,
            self.length,
        ) = _split_data(
            self.X, self.y, self.length, self.y, split, val_split, test_split, seed
        )
        # Remove test data if no test split
        if type(self.X_test) is not torch.Tensor and np.isnan(self.X_test):
            del self.X_test, self.y_test, self.length_test

    def _pad(self, Xi, to):
        """Pad trajectories to length 'to'."""
        Xi = pad_sequence([torch.tensor(Xij) for Xij in Xi])
        out = torch.full((to, Xi.size(1)), float("nan"))  # shape (s, c)
        out[0 : Xi.size(0)] = Xi  # add trajectories
        return out

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx], "length": self.length[idx]}


class PhysioNet2019(Dataset):
    """
    Creates and returns the PhysioNet Challenge 2019
    `data <https://physionet.org/content/challenge-2019/1.0.0/>`_ as a PyTorch
    DataSet.

    Data are downloaded, unpacked and saved in the ``path`` directory (default
    ``./data/physionet2019``). The size of the training, validation and (optional)
    test splits are controlled using the ``train_split`` and ``val_split`` arguments.
    For a training/validation split use ``train_split`` only. For a
    training/validation/test split specify both ``train_split`` and ``val_split``. For
    example ``train_split = 0.8`` generates a 80/20% train/validation split and
    ``train_split = 0.7``, ``val_split = 0.2`` generates a 70/20/10%
    train/validation/test split. Splits are formed using stratified sampling.
    Randomness is controlled with the ``seed`` argument.

    The ``X``, ``y`` and ``length`` attributes correspond to the split specified by the
    ``split`` argument. However training, validation and test data are available by
    accessing the ``X_[train/val/test]``, ``y_[train/val/test]`` and
    ``length__[train/val/test]`` attributes regardless of the split specified in
    ``split``.

    The ``time``, ``mask`` and ``delta`` arguments are used to append a time stamp
    channel (added by default as the first channel), missing data mask and time delta
    channels. The time delta represents the time since the previous observation
    calculated as in `Che et al (2018) <https://doi.org/10.1038/s41598-018-24271-9>`_.

    Args:
        split: The data split to return, either 'train', 'val' (validation) or 'test'.
        train_split: Proportion of data in the training set.
        val_split: Proportion of data in the validation set (optional, see above).
        time: Add time stamp in the first channel (default True).
        mask: Add missing data mask (default False).
        delta: Add time delta (default False).
        path: Save path for downloaded data (default ``./data``).
        seed: Random seed for reproducibility (optional).

    Attributes:
        X (Tensor): A tensor of default shape (*n*, *s*, *c* + 1) where *n* = number of
            trajectories, *s* = (maximum) trajectory length and *c* = number of channels
            in the PhysioNet data (*including* the ``ICULOS`` time stamp). By default, a
            time stamp is appended as the first channel. If ``time`` is False, the
            time stamp is omitted and the tensor has shape (*n*, *s*, *c*).

            A missing data mask and/or time delta channels can be added with the
            ``mask`` and ``delta`` arguments. These each have the same number of
            channels as the PhysioNet data. For example, if ``time``, ``mask`` and
            ``delta`` are all True, ``X`` has shape (*n*, *s*, 3 * *c* + 1 = 121).

            Note that PhysioNet trajectories are of unequal length and are therefore
            padded with ``NaNs`` to the length of the longest trajectory in the data.
        y (Tensor): ``SepsisLabel`` for each time point. A tensor of shape (*n*, *s*).
        length (Tensor): Length of each trajectory prior to padding. A tensor of shape
            (*n*, 1).

    Returns:
        torch.utils.data.Dataset: A PyTorch DataSet object which can be passed to a
        DataLoader.
    """

    def __init__(
        self,
        split: str,
        train_split: float,
        val_split: float,
        time: bool = True,
        mask: bool = False,
        delta: bool = False,
        path: str = "./data/physionet2019",
        seed: int = None,
    ) -> None:
        self.DATASETS = {
            "training": "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip",  # noqa: E501
            "training_setB": "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip",  # noqa: E501
        }
        self.path = pathlib.Path() / path
        # Data splits
        if val_split is None:
            test_split = 0
            val_split = 1 - train_split
        else:
            test_split = 1 - train_split - val_split
            val_split = val_split / (1 - test_split)
        # Check inputs
        splits = ["train", "val"]
        if test_split > np.finfo(float).eps:
            splits.append("test")
        assert split in splits, "argument 'split' must be one of {}".format(splits)
        # Get data
        if not os.path.isdir(self.path) or set(os.listdir(self.path)) & set(
            self.DATASETS.keys()
        ) != set(self.DATASETS.keys()):
            [self._download_zip(url) for url in self.DATASETS.values()]
        # Form tensors
        all_data = [
            self._process_set(self.path / dataset) for dataset in self.DATASETS.keys()
        ]
        self.X = torch.cat([X for X, _ in all_data])  # may fail if data is updated
        self.length = torch.cat([length for _, length in all_data])
        self.y = self.X[:, :, -1]
        self.X = self.X[:, :, :-1]
        # Add time stamp/mask/time delta channels
        if time:
            self.X = _add_time(self.X)
        if mask:
            self.X = _add_mask(self.X, time)
        if delta:
            self.X = _add_time_delta(self.X, time, mask)
        # Form test/train/validation splits
        stratify = torch.nansum(self.y, dim=1) > 0
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
            self.X,
            self.y,
            self.length,
        ) = _split_data(
            self.X,
            self.y,
            self.length,
            stratify,
            split,
            val_split,
            test_split,
            seed,
        )
        # Remove test data if no test split
        if type(self.X_test) is not torch.Tensor and np.isnan(self.X_test):
            del self.X_test, self.y_test, self.length_test

    def _download_zip(self, url):
        """Download and extract a .zip file from 'url'."""
        temp_file = tempfile.NamedTemporaryFile()
        download = requests.get(url)
        with open(temp_file.name, "wb") as file:
            file.write(download.content)
        with zipfile.ZipFile(temp_file.name, "r") as f:
            f.extractall(self.path)
        temp_file.close()

    def _process_set(self, path):
        """Process '.psv' files."""
        # Get n = number of trajectories, s = longest trajectory, c = number of
        # channels from data
        n, s, c = len([file for file in os.listdir(path)]), [], []
        for filename in os.listdir(path):
            with open(path / filename) as file:
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
        for i, filename in enumerate(os.listdir(path)):
            with open(path / filename) as file:
                reader = csv.reader(file, delimiter="|")
                for j, Xij in enumerate(reader):
                    if j > 0:
                        X[i, j - 1] = Xij
        return torch.tensor(X), torch.tensor(s)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx], "length": self.length[idx]}
