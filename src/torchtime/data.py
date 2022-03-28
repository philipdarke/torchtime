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


class UEA(Dataset):
    """
    Creates and returns a time series classification data set from the UEA/UCR
    repository as a PyTorch DataSet. See http://www.timeseriesclassification.com for
    more details and the available data sets.

    Data are downloaded using the ``sktime`` package. The size of the training,
    validation and (optional) test splits are controlled with the ``train_split`` and
    ``val_split`` arguments. For example ``train_split = 0.7``, ``val_split = 0.2``
    generates a 70/20/10% train/validation/test split, whilst ``train_split = 0.8``,
    ``val_split = 0.2`` generates a 80/20% train/validation split. Splits are formed
    using stratified sampling.

    Missing data can be simulated using the ``missing`` argument. Randomness is
    controlled with the ``seed`` argument.

    The ``X``, ``y`` and ``length`` attributes correspond to the split specified by the
    ``split`` argument. However training, validation and test data are available by
    accessing the ``X_[train/val/test]``, ``y_[train/val/test]`` and
    ``length__[train/val/test]`` attributes regardless of the split specified in
    ``split``.

    Args:
        dataset: The data set to return.
        split: The data split to return, either 'train', 'val' (validation) or 'test'.
        train_split: Proportion of data in the training set.
        val_split: Proportion of data in the validation set.
        missing: The proportion of data to drop at random. If ``missing`` is a single
            value, data is dropped from all channels. To drop data at different rates
            for each channel, pass a list of rates with the proportion missing for each
            channel (default 0).
        seed: Random seed for reproducibility (optional).

    Attributes:
        X (Tensor): A tensor of shape (*n*, *s*, *c*) where *n* = number of
            trajectories, *s* = (maximum) trajectory length and *c* = number of channels
            plus 1. The first channel is a time stamp. The remaining channels are as
            documented in the UEA/UCR repository for the data set. Where trajectories
            are of unequal lengths they are padded with ``NaNs`` to the length of the
            longest trajectory in the data.
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
        val_split: float,
        missing: Union[float, List[float]] = 0,
        seed: int = None,
    ) -> None:
        # Class variables
        self.missing = missing
        self.test_split = 1 - train_split - val_split
        self.val_split = val_split / (1 - self.test_split)
        self.seed = seed
        # Check arguments
        splits = ["train", "val"]
        if self.test_split > np.finfo(float).eps:
            splits.append("test")
        assert split in splits, "argument 'split' must be one of {}".format(splits)
        if type(self.missing) not in [int, float, list]:
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
        generator = torch.Generator()
        if self.seed is not None:
            generator = generator.manual_seed(self.seed)
        for Xi in self.X:
            if type(self.missing) in [int, float]:
                n_samples = int(max_length * self.missing)
                mask = torch.randperm(max_length, generator=generator)[:n_samples]
                Xi[mask, 1:] = float("nan")
            else:
                assert (
                    Xi.size(-1) == len(self.missing) + 1
                ), "argument 'missing' must be same length as number of channels \
                    ({})".format(
                    Xi.size(-1) - 1
                )
                for channel, rate in enumerate(self.missing):
                    n_samples = int(max_length * rate)
                    mask = torch.randperm(max_length, generator=generator)[:n_samples]
                    Xi[mask, channel + 1] = float("nan")
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
        ) = self._split_data(self.X, self.y, self.length)
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

    def _pad(self, Xi, to):
        """Pad trajectories to length 'to' and add time channel."""
        Xi = pad_sequence([torch.tensor(Xij) for Xij in Xi])
        out = torch.full((to, Xi.size(1) + 1), float("nan"))  # shape (s, c+1)
        out[:, 0] = torch.arange(0, to)  # channel 1 is time
        out[0 : Xi.size(0), 1:] = Xi  # add trajectories
        return out

    def _split_data(self, X, y, length):
        """Split data into test, training and validation sets using stratified
        sampling."""
        strat = y
        if self.test_split > np.finfo(float).eps:
            # Test split
            test_nontest_split = train_test_split(
                X,
                y,
                length,
                strat,
                train_size=self.test_split,
                random_state=self.seed,
                shuffle=True,
                stratify=strat,
            )
            (
                X_test,
                X_nontest,
                y_test,
                y_nontest,
                length_test,
                length_nontest,
                _,
                strat_nontest,
            ) = test_nontest_split
            # Validation/train split
            val_train_split = train_test_split(
                X_nontest,
                y_nontest,
                length_nontest,
                train_size=self.val_split,
                random_state=self.seed,
                shuffle=True,
                stratify=strat_nontest,
            )
            X_val, X_train, y_val, y_train, length_val, length_train = val_train_split
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
        else:
            # Validation/train split
            val_train_split = train_test_split(
                X,
                y,
                length,
                train_size=self.val_split,
                random_state=self.seed,
                shuffle=True,
                stratify=strat,
            )
            X_val, X_train, y_val, y_train, length_val, length_train = val_train_split
            return (
                X_train,
                y_train,
                length_train,
                X_val,
                y_val,
                length_val,
                float("nan"),
                float("nan"),
                float("nan"),
            )

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx], "length": self.length[idx]}


class PhysioNet2019(Dataset):
    """
    Creates and returns the PhysioNet Challenge 2019 data as a PyTorch DataSet. See
    https://physionet.org/content/challenge-2019/1.0.0/ for more details.

    Data are downloaded, unpacked and saved in the ``path`` directory (default
    ``./data``). The size of the training, validation and (optional) test splits are
    controlled with the ``train_split`` and ``val_split`` arguments. For example
    ``train_split = 0.7``, ``val_split = 0.2`` generates a 70/20/10%
    train/validation/test split, whilst ``train_split = 0.8``, ``val_split = 0.2``
    generates a 80/20% train/validation split. Splits are formed using stratified
    sampling. Randomness is controlled with the ``seed`` argument.

    The ``X``, ``y`` and ``length`` attributes correspond to the split specified by the
    ``split`` argument. However training, validation and test data are available by
    accessing the ``X_[train/val/test]``, ``y_[train/val/test]`` and
    ``length__[train/val/test]`` attributes regardless of the split specified in
    ``split``.

    Args:
        split: The data split to return, either 'train', 'val' (validation) or 'test'.
        train_split: Proportion of data in the training set.
        val_split: Proportion of data in the validation set.
        path: Save path for downloaded data (default ``./data``).
        seed: Random seed for reproducibility (optional).

    Attributes:
        X (Tensor): A tensor of shape (*n*, *s*, *c*) where *n* = number of
            trajectories, *s* = (maximum) trajectory length and *c* = number of channels
            plus 1. The first channel is a time stamp. The remaining channels are the
            PhysioNet data (*including* the ``ICULOS`` time stamp). Note that PhysioNet
            trajectories are of unequal length and are therefore padded with ``NaNs`` to
            the length of the longest trajectory in the data.
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
        path: str = "./data",
        seed: int = None,
    ) -> None:
        self.DATASETS = {
            "training": "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip",  # noqa: E501
            "training_setB": "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip",  # noqa: E501
        }
        self.test_split = 1 - train_split - val_split
        self.val_split = val_split / (1 - self.test_split)
        self.path = pathlib.Path() / path
        self.seed = seed
        # Check inputs
        splits = ["train", "val"]
        if self.test_split > np.finfo(float).eps:
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
        time_stamp = torch.arange(self.X.size(1)).unsqueeze(0)
        time_stamp = time_stamp.tile((self.X.size(0), 1)).unsqueeze(2)
        self.X = torch.cat([time_stamp, self.X[:, :, 0:-1]], dim=2)  # add time stamp
        # Test/train/validation splits
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
        ) = self._split_data(self.X, self.y, self.length)
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

    def _split_data(self, X, y, length):
        """Split data into test, training and validation sets using stratified
        sampling."""
        strat = torch.nansum(y, dim=1) > 0
        if self.test_split > np.finfo(float).eps:
            # Test split
            test_nontest_split = train_test_split(
                X,
                y,
                length,
                strat,
                train_size=self.test_split,
                random_state=self.seed,
                shuffle=True,
                stratify=strat,
            )
            (
                X_test,
                X_nontest,
                y_test,
                y_nontest,
                length_test,
                length_nontest,
                _,
                strat_nontest,
            ) = test_nontest_split
            # Validation/train split
            val_train_split = train_test_split(
                X_nontest,
                y_nontest,
                length_nontest,
                train_size=self.val_split,
                random_state=self.seed,
                shuffle=True,
                stratify=strat_nontest,
            )
            X_val, X_train, y_val, y_train, length_val, length_train = val_train_split
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
        else:
            # Validation/train split
            val_train_split = train_test_split(
                X,
                y,
                length,
                train_size=self.val_split,
                random_state=self.seed,
                shuffle=True,
                stratify=strat,
            )
            X_val, X_train, y_val, y_train, length_val, length_train = val_train_split
            return (
                X_train,
                y_train,
                length_train,
                X_val,
                y_val,
                length_val,
                float("nan"),
                float("nan"),
                float("nan"),
            )

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx], "length": self.length[idx]}
