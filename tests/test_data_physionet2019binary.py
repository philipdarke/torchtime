import re

import pytest
import torch

from torchtime.constants import OBJ_EXT
from torchtime.data import PhysioNet2019Binary
from torchtime.utils import _get_SHA256

SEED = 456789
RTOL = 1e-4
ATOL = 1e-4


class TestPhysioNet2019Binary:
    """Test PhysioNet2019Binary class."""

    def test_invalid_split_arg(self):
        """Catch invalid split argument."""
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'split' must be one of ['train', 'val']"),
        ):
            PhysioNet2019Binary(
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
            PhysioNet2019Binary(
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
            PhysioNet2019Binary(
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
            PhysioNet2019Binary(
                split="test",
                train_prop=new_prop,
                val_prop=new_prop,
                seed=SEED,
            )

    def test_load_data(self):
        """Validate data set."""
        PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            seed=SEED,
        )
        assert (
            _get_SHA256(".torchtime/physionet_2019binary/X" + OBJ_EXT)
            == "52b28760d1d2420b27d8003531ae42cdca14c6e408b024273c81a65d39c2f2df"
        )
        assert (
            _get_SHA256(".torchtime/physionet_2019binary/y" + OBJ_EXT)
            == "dd78b55b728fe62798cadb28741d5cb0f243dfb3146aa6732baa26ecfe32ba40"
        )
        assert (
            _get_SHA256(".torchtime/physionet_2019binary/length" + OBJ_EXT)
            == "d0c1c809d47485cb237e650d11264bcc9225f2c56d5754d4ebeda41cc87c63ba"
        )

    def test_train_val(self):
        """Test training/validation split sizes."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28234, 72, 41])
        assert dataset.y_train.shape == torch.Size([28234, 1])
        assert dataset.length_train.shape == torch.Size([28234])
        assert dataset.X_val.shape == torch.Size([12099, 72, 41])
        assert dataset.y_val.shape == torch.Size([12099, 1])
        assert dataset.length_val.shape == torch.Size([12099])
        # Ensure no test data is returned
        with pytest.raises(
            AttributeError,
            match=re.escape("'PhysioNet2019Binary' object has no attribute 'X_test'"),
        ):
            dataset.X_test
        with pytest.raises(
            AttributeError,
            match=re.escape("'PhysioNet2019Binary' object has no attribute 'y_test'"),
        ):
            dataset.y_test
        with pytest.raises(
            AttributeError,
            match=re.escape(
                "'PhysioNet2019Binary' object has no attribute 'length_test'"
            ),
        ):
            dataset.length_test

    def test_train_val_test(self):
        """Test training/validation/test split sizes."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28234, 72, 41])
        assert dataset.y_train.shape == torch.Size([28234, 1])
        assert dataset.length_train.shape == torch.Size([28234])
        assert dataset.X_val.shape == torch.Size([8066, 72, 41])
        assert dataset.y_val.shape == torch.Size([8066, 1])
        assert dataset.length_val.shape == torch.Size([8066])
        assert dataset.X_test.shape == torch.Size([4033, 72, 41])
        assert dataset.y_test.shape == torch.Size([4033, 1])
        assert dataset.length_test.shape == torch.Size([4033])

    def test_train_split(self):
        """Test training split is returned."""
        dataset = PhysioNet2019Binary(
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
        dataset = PhysioNet2019Binary(
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
        dataset = PhysioNet2019Binary(
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
        dataset = PhysioNet2019Binary(
            split="test",
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

    def test_invalid_impute(self):
        """Catch invalid impute arguments."""
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "argument 'impute' must be a string in ['none', 'mean', 'forward'] or a function"  # noqa: E501
            ),
        ):
            PhysioNet2019Binary(
                split="train",
                train_prop=0.7,
                impute="blah",
                seed=SEED,
            )
        with pytest.raises(
            Exception,
            match=re.escape(
                "argument 'impute' must be a string in ['none', 'mean', 'forward'] or a function"  # noqa: E501
            ),
        ):
            PhysioNet2019Binary(
                split="train",
                train_prop=0.7,
                impute=3,
                seed=SEED,
            )

    def test_no_impute(self):
        """Test no imputation."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute="none",
            seed=SEED,
        )
        # Check number of NaNs
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 68853366
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 19690045
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 9858312
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_mean_impute(self):
        """Test mean imputation."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute="mean",
            seed=SEED,
        )
        # Check no NaNs post imputation
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_forward_impute(self):
        """Test forward imputation."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute="forward",
            seed=SEED,
        )
        # Check no NaNs post imputation
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_custom_imputation_1(self):
        """Test custom imputation function."""

        def impute_with_zero(X, y, fill, select):
            return X.nan_to_num(0), y.nan_to_num(0)

        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute=impute_with_zero,
            seed=SEED,
        )
        # Check number of NaNs
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_custom_imputation_2(self):
        """Test custom imputation function."""

        def no_imputation(X, y, fill, select):
            return X, y

        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute=no_imputation,
            seed=SEED,
        )
        # Check number of NaNs
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 68853366
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 19690045
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 9858312
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_time(self):
        """Test time argument."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28234, 72, 41])
        assert dataset.X_val.shape == torch.Size([8066, 72, 41])
        assert dataset.X_test.shape == torch.Size([4033, 72, 41])
        # Check time channel
        for i in range(72):
            assert torch.equal(
                dataset.X_train[:, i, 0],
                torch.full([28234], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 0],
                torch.full([8066], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 0],
                torch.full([4033], fill_value=i, dtype=torch.float),
            )

    def test_no_time(self):
        """Test time argument."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28234, 72, 40])
        assert dataset.X_val.shape == torch.Size([8066, 72, 40])
        assert dataset.X_test.shape == torch.Size([4033, 72, 40])

    def test_mask(self):
        """Test mask argument."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            mask=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28234, 72, 80])
        assert dataset.X_val.shape == torch.Size([8066, 72, 80])
        assert dataset.X_test.shape == torch.Size([4033, 72, 80])

    def test_delta(self):
        """Test time delta argument."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            delta=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28234, 72, 80])
        assert dataset.X_val.shape == torch.Size([8066, 72, 80])
        assert dataset.X_test.shape == torch.Size([4033, 72, 80])
        # Check time delta channel
        assert torch.equal(
            dataset.X_train[:, 0, 40], torch.zeros([28234], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_val[:, 0, 40], torch.zeros([8066], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_test[:, 0, 40], torch.zeros([4033], dtype=torch.float)
        )

    def test_time_mask_delta(self):
        """Test combination of time/mask/delta arguments."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            mask=True,
            delta=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28234, 72, 121])
        assert dataset.X_val.shape == torch.Size([8066, 72, 121])
        assert dataset.X_test.shape == torch.Size([4033, 72, 121])
        # Check time channel
        for i in range(72):
            assert torch.equal(
                dataset.X_train[:, i, 0],
                torch.full([28234], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 0],
                torch.full([8066], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 0],
                torch.full([4033], fill_value=i, dtype=torch.float),
            )
        # Check time delta channel
        assert torch.equal(
            dataset.X_train[:, 0, 81], torch.zeros([28234], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_val[:, 0, 81], torch.zeros([8066], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_test[:, 0, 81], torch.zeros([4033], dtype=torch.float)
        )

    def test_standarisation(self):
        """Check training data is standardised."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            time=False,
            standardise=True,
            seed=SEED,
        )
        for c, Xc in enumerate(dataset.X_train.unbind(dim=-1)):
            assert torch.allclose(
                torch.nanmean(Xc), torch.Tensor([0.0]), rtol=RTOL, atol=ATOL
            )
            assert torch.allclose(
                torch.std(Xc[~torch.isnan(Xc)]),
                torch.Tensor([1.0]),
                rtol=RTOL,
                atol=ATOL,
            )

    def test_reproducibility_1(self):
        """Test seed argument."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check first value in 39th channel
        assert torch.allclose(
            dataset.X_train[0, 0, 39], torch.tensor(-0.03), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 0, 39], torch.tensor(-27.55), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 0, 39], torch.tensor(-0.66), rtol=RTOL, atol=ATOL
        )

    def test_reproducibility_2(self):
        """Test seed argument."""
        dataset = PhysioNet2019Binary(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=999999,
        )
        # Check first value in 39th channel
        assert torch.allclose(
            dataset.X_train[0, 0, 39], torch.tensor(-0.01), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 0, 39], torch.tensor(-90.73), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 0, 39], torch.tensor(-199.47), rtol=RTOL, atol=ATOL
        )
