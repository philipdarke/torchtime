import re

import pytest
import torch

from torchtime.constants import OBJ_EXT
from torchtime.data import UEA
from torchtime.utils import _get_SHA256

DATASET = "CharacterTrajectories"
SEED = 456789
RTOL = 1e-4
ATOL = 1e-4
SHA_X = "fb5077e0da758b241caff0270c28e1a3b2ac0ea2aebc4e5b2276b89947a5d257"
SHA_Y = "59b53905692701ec7e6c5ca2e8b61d06c44ec531a3d29c0bbb38ddec7754da80"
SHA_LENGTH = "698d1ebefff52cff3c41da20feccbd141e856b16aa618b7e44c80daf54ab0e39"
N_DATA_CHANNELS = 3
N_CLASSES = 20


class TestUEACharacterTrajectories:
    """Test UEA class with CharacterTrajectories data set."""

    def test_invalid_split_arg(self):
        """Catch invalid split argument."""
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'split' must be one of ['train', 'val']"),
        ):
            UEA(
                dataset=DATASET,
                split="xyz",
                train_prop=0.8,
                seed=SEED,
            )

    def test_invalid_split_size(self):
        """Catch invalid split sizes."""
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'train_prop' must be in range (0, 1)"),
        ):
            UEA(
                dataset=DATASET,
                split="train",
                train_prop=-0.5,
                seed=SEED,
            )

    def test_incompatible_split_size(self):
        """Catch incompatible split sizes."""
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'train_prop' must be in range (0, 1)"),
        ):
            UEA(
                dataset=DATASET,
                split="train",
                train_prop=1,
                seed=SEED,
            )
        new_prop = 0.5
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "argument 'val_prop' must be in range (0, {})".format(1 - new_prop)
            ),
        ):
            UEA(
                dataset=DATASET,
                split="test",
                train_prop=new_prop,
                val_prop=new_prop,
                seed=SEED,
            )

    def test_load_data(self):
        """Validate data set."""
        UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            seed=SEED,
        )
        assert _get_SHA256(".torchtime/uea_" + DATASET + "/X" + OBJ_EXT) == SHA_X
        assert _get_SHA256(".torchtime/uea_" + DATASET + "/y" + OBJ_EXT) == SHA_Y
        assert (
            _get_SHA256(".torchtime/uea_" + DATASET + "/length" + OBJ_EXT) == SHA_LENGTH
        )

    def test_train_val(self):
        """Test training/validation split sizes."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([2001, 182, N_DATA_CHANNELS + 1])
        assert dataset.y_train.shape == torch.Size([2001, N_CLASSES])
        assert dataset.length_train.shape == torch.Size([2001])
        assert dataset.X_val.shape == torch.Size([857, 182, N_DATA_CHANNELS + 1])
        assert dataset.y_val.shape == torch.Size([857, N_CLASSES])
        assert dataset.length_val.shape == torch.Size([857])
        # Ensure no test data is returned
        with pytest.raises(
            AttributeError, match=re.escape("'UEA' object has no attribute 'X_test'")
        ):
            dataset.X_test
        with pytest.raises(
            AttributeError, match=re.escape("'UEA' object has no attribute 'y_test'")
        ):
            dataset.y_test
        with pytest.raises(
            AttributeError,
            match=re.escape("'UEA' object has no attribute 'length_test'"),
        ):
            dataset.length_test

    def test_train_val_test(self):
        """Test training/validation/test split sizes."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([2002, 182, N_DATA_CHANNELS + 1])
        assert dataset.y_train.shape == torch.Size([2002, N_CLASSES])
        assert dataset.length_train.shape == torch.Size([2002])
        assert dataset.X_val.shape == torch.Size([571, 182, N_DATA_CHANNELS + 1])
        assert dataset.y_val.shape == torch.Size([571, N_CLASSES])
        assert dataset.length_val.shape == torch.Size([571])
        assert dataset.X_test.shape == torch.Size([285, 182, N_DATA_CHANNELS + 1])
        assert dataset.y_test.shape == torch.Size([285, N_CLASSES])
        assert dataset.length_test.shape == torch.Size([285])

    def test_train_split(self):
        """Test training split is returned."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check correct split is returned
        assert torch.allclose(dataset.X, dataset.X_train, equal_nan=True)
        assert torch.allclose(dataset.y, dataset.y_train, equal_nan=True)
        assert torch.allclose(dataset.length, dataset.length_train, equal_nan=True)

    def test_val_split(self):
        """Test validation split is returned."""
        dataset = UEA(
            dataset=DATASET,
            split="val",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check correct split is returned
        assert torch.allclose(dataset.X, dataset.X_val, equal_nan=True)
        assert torch.allclose(dataset.y, dataset.y_val, equal_nan=True)
        assert torch.allclose(dataset.length, dataset.length_val, equal_nan=True)

    def test_test_split(self):
        """Test test split is returned."""
        dataset = UEA(
            dataset=DATASET,
            split="test",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check correct split is returned
        assert torch.allclose(dataset.X, dataset.X_test, equal_nan=True)
        assert torch.allclose(dataset.y, dataset.y_test, equal_nan=True)
        assert torch.allclose(dataset.length, dataset.length_test, equal_nan=True)

    def test_length(self):
        """Test length attribute."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            seed=SEED,
        )
        for i, Xi in enumerate(dataset.X_train.unbind()):
            length_i = dataset.length_train[i]
            assert not torch.all(torch.isnan(Xi[length_i - 1]))
            assert torch.all(torch.isnan(Xi[length_i:]))
        for i, Xi in enumerate(dataset.X_val.unbind()):
            length_i = dataset.length_val[i]
            assert not torch.all(torch.isnan(Xi[length_i - 1]))
            assert torch.all(torch.isnan(Xi[length_i:]))
        for i, Xi in enumerate(dataset.X_test.unbind()):
            length_i = dataset.length_test[i]
            assert not torch.all(torch.isnan(Xi[length_i - 1]))
            assert torch.all(torch.isnan(Xi[length_i:]))

    def test_missing(self):
        """Test missing data simulation."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            missing=0.5,
            seed=SEED,
        )
        # Check number of NaNs
        assert (
            torch.sum(torch.isnan(dataset.X_train[:, :, dataset.data_idx])).item()
            == 732969
        )  # expect around 3 * (239873 * 0.5 + 2002 * 182 - 239873) = 733,283
        assert (
            torch.sum(torch.isnan(dataset.X_val[:, :, dataset.data_idx])).item()
            == 208686
        )  # expect around 3 * (68797 * 0.5 + 571 * 182 - 68797) = 208,571
        assert (
            torch.sum(torch.isnan(dataset.X_test[:, :, dataset.data_idx])).item()
            == 103872
        )  # expect around 3 * (34269 * 0.5 + 285 * 182 - 34269) = 104,207

    def test_invalid_impute(self):
        """Catch invalid impute arguments."""
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "argument 'impute' must be a string in ['none', 'zero', 'mean', 'forward'] or a function"  # noqa: E501
            ),
        ):
            UEA(
                dataset=DATASET,
                split="train",
                train_prop=0.7,
                missing=0.5,
                impute="blah",
                seed=SEED,
            )
        with pytest.raises(
            Exception,
            match=re.escape(
                "argument 'impute' must be a string in ['none', 'zero', 'mean', 'forward'] or a function"  # noqa: E501
            ),
        ):
            UEA(
                dataset=DATASET,
                split="train",
                train_prop=0.7,
                missing=0.5,
                impute=3,
                seed=SEED,
            )

    def test_no_impute(self):
        """Test no imputation."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            missing=0.5,
            impute="none",
            seed=SEED,
        )
        # Check number of NaNs
        assert (
            torch.sum(torch.isnan(dataset.X_train[:, :, dataset.data_idx])).item()
            == 732969
        )
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert (
            torch.sum(torch.isnan(dataset.X_val[:, :, dataset.data_idx])).item()
            == 208686
        )
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert (
            torch.sum(torch.isnan(dataset.X_test[:, :, dataset.data_idx])).item()
            == 103872
        )
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_zero_impute(self):
        """Test zero imputation."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            missing=0.5,
            impute="zero",
            seed=SEED,
        )
        # Check no NaNs post imputation
        assert (
            torch.sum(torch.isnan(dataset.X_train[:, :, dataset.data_idx])).item() == 0
        )
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val[:, :, dataset.data_idx])).item() == 0
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert (
            torch.sum(torch.isnan(dataset.X_test[:, :, dataset.data_idx])).item() == 0
        )
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_mean_impute(self):
        """Test mean imputation."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            missing=0.5,
            impute="mean",
            seed=SEED,
        )
        # Check no NaNs post imputation
        assert (
            torch.sum(torch.isnan(dataset.X_train[:, :, dataset.data_idx])).item() == 0
        )
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val[:, :, dataset.data_idx])).item() == 0
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert (
            torch.sum(torch.isnan(dataset.X_test[:, :, dataset.data_idx])).item() == 0
        )
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_forward_impute(self):
        """Test forward imputation."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            missing=0.5,
            impute="forward",
            seed=SEED,
        )
        # Check no NaNs post imputation
        assert (
            torch.sum(torch.isnan(dataset.X_train[:, :, dataset.data_idx])).item() == 0
        )
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val[:, :, dataset.data_idx])).item() == 0
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert (
            torch.sum(torch.isnan(dataset.X_test[:, :, dataset.data_idx])).item() == 0
        )
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_custom_imputation_1(self):
        """Test custom imputation function."""

        def impute_with_zero(X, y, fill, select):
            return X.nan_to_num(0), y.nan_to_num(0)

        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            missing=0.5,
            impute=impute_with_zero,
            seed=SEED,
        )
        # Check number of NaNs
        assert (
            torch.sum(torch.isnan(dataset.X_train[:, :, dataset.data_idx])).item() == 0
        )
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val[:, :, dataset.data_idx])).item() == 0
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert (
            torch.sum(torch.isnan(dataset.X_test[:, :, dataset.data_idx])).item() == 0
        )
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_custom_imputation_2(self):
        """Test custom imputation function."""

        def no_imputation(X, y, fill, select):
            return X, y

        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            missing=0.5,
            impute=no_imputation,
            seed=SEED,
        )
        # Check number of NaNs
        assert (
            torch.sum(torch.isnan(dataset.X_train[:, :, dataset.data_idx])).item()
            == 732969
        )
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert (
            torch.sum(torch.isnan(dataset.X_val[:, :, dataset.data_idx])).item()
            == 208686
        )
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert (
            torch.sum(torch.isnan(dataset.X_test[:, :, dataset.data_idx])).item()
            == 103872
        )
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_overwrite_data(self):
        """Overwrite cache and validate data set."""
        UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            seed=SEED,
            overwrite_cache=True,
        )
        assert _get_SHA256(".torchtime/uea_" + DATASET + "/X" + OBJ_EXT) == SHA_X
        assert _get_SHA256(".torchtime/uea_" + DATASET + "/y" + OBJ_EXT) == SHA_Y
        assert (
            _get_SHA256(".torchtime/uea_" + DATASET + "/length" + OBJ_EXT) == SHA_LENGTH
        )

    def test_time(self):
        """Test time argument."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([2002, 182, N_DATA_CHANNELS + 1])
        assert dataset.X_val.shape == torch.Size([571, 182, N_DATA_CHANNELS + 1])
        assert dataset.X_test.shape == torch.Size([285, 182, N_DATA_CHANNELS + 1])
        # Check time channels
        for i, Xi in enumerate(dataset.X_train):
            assert torch.equal(
                Xi[: dataset.length_train[i], 0],
                torch.arange(dataset.length_train[i], dtype=torch.float),
            )
        for i, Xi in enumerate(dataset.X_val):
            assert torch.equal(
                Xi[: dataset.length_val[i], 0],
                torch.arange(dataset.length_val[i], dtype=torch.float),
            )
        for i, Xi in enumerate(dataset.X_test):
            assert torch.equal(
                Xi[: dataset.length_test[i], 0],
                torch.arange(dataset.length_test[i], dtype=torch.float),
            )

    def test_no_time(self):
        """Test time argument."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([2002, 182, N_DATA_CHANNELS])
        assert dataset.X_val.shape == torch.Size([571, 182, N_DATA_CHANNELS])
        assert dataset.X_test.shape == torch.Size([285, 182, N_DATA_CHANNELS])

    def test_mask(self):
        """Test mask argument."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            mask=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([2002, 182, 2 * N_DATA_CHANNELS])
        assert dataset.X_val.shape == torch.Size([571, 182, 2 * N_DATA_CHANNELS])
        assert dataset.X_test.shape == torch.Size([285, 182, 2 * N_DATA_CHANNELS])
        # Check mask channels
        max_sequence = int(dataset.X.size(1))
        for i, Xi in enumerate(dataset.X_train):
            mask = torch.zeros((max_sequence, N_DATA_CHANNELS), dtype=torch.float)
            mask[: dataset.length_train[i], :] = 1
            assert torch.equal(Xi[:, N_DATA_CHANNELS:], mask)
        for i, Xi in enumerate(dataset.X_val):
            mask = torch.zeros((max_sequence, N_DATA_CHANNELS), dtype=torch.float)
            mask[: dataset.length_val[i], :] = 1
            assert torch.equal(Xi[:, N_DATA_CHANNELS:], mask)
        for i, Xi in enumerate(dataset.X_test):
            mask = torch.zeros((max_sequence, N_DATA_CHANNELS), dtype=torch.float)
            mask[: dataset.length_test[i], :] = 1
            assert torch.equal(Xi[:, N_DATA_CHANNELS:], mask)

    def test_delta(self):
        """Test time delta argument."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            delta=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([2002, 182, 2 * N_DATA_CHANNELS])
        assert dataset.X_val.shape == torch.Size([571, 182, 2 * N_DATA_CHANNELS])
        assert dataset.X_test.shape == torch.Size([285, 182, 2 * N_DATA_CHANNELS])
        # Check time delta channels
        for i, Xi in enumerate(dataset.X_train):
            delta = torch.ones(
                (dataset.length_train[i], N_DATA_CHANNELS), dtype=torch.float
            )
            delta[0, :] = 0
            assert torch.equal(Xi[: dataset.length_train[i], N_DATA_CHANNELS:], delta)
        for i, Xi in enumerate(dataset.X_val):
            delta = torch.ones(
                (dataset.length_val[i], N_DATA_CHANNELS), dtype=torch.float
            )
            delta[0, :] = 0
            assert torch.equal(Xi[: dataset.length_val[i], N_DATA_CHANNELS:], delta)
        for i, Xi in enumerate(dataset.X_test):
            delta = torch.ones(
                (dataset.length_test[i], N_DATA_CHANNELS), dtype=torch.float
            )
            delta[0, :] = 0
            assert torch.equal(Xi[: dataset.length_test[i], N_DATA_CHANNELS:], delta)

    def test_time_mask_delta(self):
        """Test combination of time/mask/delta arguments."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            mask=True,
            delta=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([2002, 182, 3 * N_DATA_CHANNELS + 1])
        assert dataset.X_val.shape == torch.Size([571, 182, 3 * N_DATA_CHANNELS + 1])
        assert dataset.X_test.shape == torch.Size([285, 182, 3 * N_DATA_CHANNELS + 1])
        # Check time channel
        for i, Xi in enumerate(dataset.X_train):
            assert torch.equal(
                Xi[: dataset.length_train[i], 0],
                torch.arange(dataset.length_train[i], dtype=torch.float),
            )
        for i, Xi in enumerate(dataset.X_val):
            assert torch.equal(
                Xi[: dataset.length_val[i], 0],
                torch.arange(dataset.length_val[i], dtype=torch.float),
            )
        for i, Xi in enumerate(dataset.X_test):
            assert torch.equal(
                Xi[: dataset.length_test[i], 0],
                torch.arange(dataset.length_test[i], dtype=torch.float),
            )
        # Check mask channels
        max_sequence = int(dataset.X.size(1))
        for i, Xi in enumerate(dataset.X_train):
            mask = torch.zeros((max_sequence, N_DATA_CHANNELS), dtype=torch.float)
            mask[: dataset.length_train[i], :] = 1
            assert torch.equal(
                Xi[:, (N_DATA_CHANNELS + 1) : (2 * N_DATA_CHANNELS + 1)], mask
            )
        for i, Xi in enumerate(dataset.X_val):
            mask = torch.zeros((max_sequence, N_DATA_CHANNELS), dtype=torch.float)
            mask[: dataset.length_val[i], :] = 1
            assert torch.equal(
                Xi[:, (N_DATA_CHANNELS + 1) : (2 * N_DATA_CHANNELS + 1)], mask
            )
        for i, Xi in enumerate(dataset.X_test):
            mask = torch.zeros((max_sequence, N_DATA_CHANNELS), dtype=torch.float)
            mask[: dataset.length_test[i], :] = 1
            assert torch.equal(
                Xi[:, (N_DATA_CHANNELS + 1) : (2 * N_DATA_CHANNELS + 1)], mask
            )
        # Check time delta channels
        for i, Xi in enumerate(dataset.X_train):
            delta = torch.ones(
                (dataset.length_train[i], N_DATA_CHANNELS), dtype=torch.float
            )
            delta[0, :] = 0
            assert torch.equal(
                Xi[: dataset.length_train[i], (2 * N_DATA_CHANNELS + 1) :], delta
            )
        for i, Xi in enumerate(dataset.X_val):
            delta = torch.ones(
                (dataset.length_val[i], N_DATA_CHANNELS), dtype=torch.float
            )
            delta[0, :] = 0
            assert torch.equal(
                Xi[: dataset.length_val[i], (2 * N_DATA_CHANNELS + 1) :], delta
            )
        for i, Xi in enumerate(dataset.X_test):
            delta = torch.ones(
                (dataset.length_test[i], N_DATA_CHANNELS), dtype=torch.float
            )
            delta[0, :] = 0
            assert torch.equal(
                Xi[: dataset.length_test[i], (2 * N_DATA_CHANNELS + 1) :], delta
            )

    def test_standarisation_1(self):
        """Check training data is standardised."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            standardise="all",
            seed=SEED,
        )
        for Xc in dataset.X_train.unbind(dim=-1):
            assert torch.allclose(
                torch.nanmean(Xc), torch.Tensor([0.0]), rtol=RTOL, atol=ATOL
            )
            assert torch.allclose(
                torch.std(Xc[~torch.isnan(Xc)]),
                torch.Tensor([1.0]),
                rtol=RTOL,
                atol=ATOL,
            ) or torch.allclose(
                torch.std(Xc[~torch.isnan(Xc)]),
                torch.Tensor([0.0]),  # if all values are the same
                rtol=RTOL,
                atol=ATOL,
            )

    def test_standarisation_2(self):
        """Check imputed training data is standardised."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            impute="forward",
            standardise="all",
            seed=SEED,
        )
        for Xc in dataset.X_train.unbind(dim=-1):
            assert torch.allclose(
                torch.nanmean(Xc), torch.Tensor([0.0]), rtol=RTOL, atol=ATOL
            )
            assert torch.allclose(
                torch.std(Xc[~torch.isnan(Xc)]),
                torch.Tensor([1.0]),
                rtol=RTOL,
                atol=ATOL,
            ) or torch.allclose(
                torch.std(Xc[~torch.isnan(Xc)]),
                torch.Tensor([0.0]),  # if all values are the same
                rtol=RTOL,
                atol=ATOL,
            )

    def test_standarisation_3(self):
        """Check training data is standardised (time series only)."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            delta=True,
            standardise="data",
            seed=SEED,
        )
        for Xc in dataset.X_train[:, :, dataset.data_idx].unbind(dim=-1):
            assert torch.allclose(
                torch.nanmean(Xc), torch.Tensor([0.0]), rtol=RTOL, atol=ATOL
            )
            assert torch.allclose(
                torch.std(Xc[~torch.isnan(Xc)]),
                torch.Tensor([1.0]),
                rtol=RTOL,
                atol=ATOL,
            ) or torch.allclose(
                torch.std(Xc[~torch.isnan(Xc)]),
                torch.Tensor([0.0]),  # if all values are the same
                rtol=RTOL,
                atol=ATOL,
            )

    def test_standarisation_4(self):
        """Check imputed training data is standardised (time series only)."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            delta=True,
            impute="forward",
            standardise="data",
            seed=SEED,
        )
        for Xc in dataset.X_train[:, :, dataset.data_idx].unbind(dim=-1):
            assert torch.allclose(
                torch.nanmean(Xc), torch.Tensor([0.0]), rtol=RTOL, atol=ATOL
            )
            assert torch.allclose(
                torch.std(Xc[~torch.isnan(Xc)]),
                torch.Tensor([1.0]),
                rtol=RTOL,
                atol=ATOL,
            ) or torch.allclose(
                torch.std(Xc[~torch.isnan(Xc)]),
                torch.Tensor([0.0]),  # if all values are the same
                rtol=RTOL,
                atol=ATOL,
            )

    def test_reproducibility_1(self):
        """Test seed argument."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check first value in each data set
        assert torch.allclose(
            dataset.X_train[0, 0, 1], torch.tensor(-0.1869), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 0, 1], torch.tensor(0.0418), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 0, 1], torch.tensor(0.0940), rtol=RTOL, atol=ATOL
        )

    def test_reproducibility_2(self):
        """Test seed argument."""
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=999999,
        )
        # Check first value in each data set
        assert torch.allclose(
            dataset.X_train[0, 0, 1], torch.tensor(0.0391), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 0, 1], torch.tensor(0.0), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 0, 1], torch.tensor(-0.0767), rtol=RTOL, atol=ATOL
        )
