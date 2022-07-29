import re

import pytest
import torch

from torchtime.constants import OBJ_EXT
from torchtime.data import PhysioNet2012
from torchtime.utils import _get_SHA256

SEED = 456789
RTOL = 1e-4
ATOL = 1e-4


class TestPhysioNet2012:
    """Test PhysioNet2012 class."""

    def test_invalid_split_arg(self):
        """Catch invalid split argument."""
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'split' must be one of ['train', 'val']"),
        ):
            PhysioNet2012(
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
            PhysioNet2012(
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
            PhysioNet2012(
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
            PhysioNet2012(
                split="test",
                train_prop=new_prop,
                val_prop=new_prop,
                seed=SEED,
            )

    def test_load_data(self):
        """Validate data set."""
        PhysioNet2012(
            split="train",
            train_prop=0.7,
            seed=SEED,
        )
        assert (
            _get_SHA256(".torchtime/physionet_2012/X" + OBJ_EXT)
            == "5c43fe40228ec6bd0122661e9ac48ad7cbdaf0778970cf9f81a5c3ac5ff67ab5"
        )
        assert (
            _get_SHA256(".torchtime/physionet_2012/y" + OBJ_EXT)
            == "5b9bf1f58ff02e04397f68ae776fc519e20cae6e66a632b01fa309693c3de3e9"
        )
        assert (
            _get_SHA256(".torchtime/physionet_2012/length" + OBJ_EXT)
            == "d4dbf3d19e9f03618f3113c57c5950031c22bad75c80744438e3121b1cff2204"
        )

    def test_train_val(self):
        """Test training/validation split sizes."""
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([8400, 215, 43])
        assert dataset.y_train.shape == torch.Size([8400, 1])
        assert dataset.length_train.shape == torch.Size([8400])
        assert dataset.X_val.shape == torch.Size([3600, 215, 43])
        assert dataset.y_val.shape == torch.Size([3600, 1])
        assert dataset.length_val.shape == torch.Size([3600])
        # Ensure no test data is returned
        with pytest.raises(
            AttributeError,
            match=re.escape("'PhysioNet2012' object has no attribute 'X_test'"),
        ):
            dataset.X_test
        with pytest.raises(
            AttributeError,
            match=re.escape("'PhysioNet2012' object has no attribute 'y_test'"),
        ):
            dataset.y_test
        with pytest.raises(
            AttributeError,
            match=re.escape("'PhysioNet2012' object has no attribute 'length_test'"),
        ):
            dataset.length_test

    def test_train_val_test(self):
        """Test training/validation/test split sizes."""
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([8400, 215, 43])
        assert dataset.y_train.shape == torch.Size([8400, 1])
        assert dataset.length_train.shape == torch.Size([8400])
        assert dataset.X_val.shape == torch.Size([2400, 215, 43])
        assert dataset.y_val.shape == torch.Size([2400, 1])
        assert dataset.length_val.shape == torch.Size([2400])
        assert dataset.X_test.shape == torch.Size([1200, 215, 43])
        assert dataset.y_test.shape == torch.Size([1200, 1])
        assert dataset.length_test.shape == torch.Size([1200])

    def test_train_split(self):
        """Test training split is returned."""
        dataset = PhysioNet2012(
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
        dataset = PhysioNet2012(
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
        dataset = PhysioNet2012(
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
        dataset = PhysioNet2012(
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
                "argument 'impute' must be a string in ['none', 'zero', 'mean', 'forward'] or a function"  # noqa: E501
            ),
        ):
            PhysioNet2012(
                split="train",
                train_prop=0.7,
                impute="blah",
                seed=SEED,
            )
        with pytest.raises(
            Exception,
            match=re.escape(
                "argument 'impute' must be a string in ['none', 'zero', 'mean', 'forward'] or a function"  # noqa: E501
            ),
        ):
            PhysioNet2012(
                split="train",
                train_prop=0.7,
                impute=3,
                seed=SEED,
            )

    def test_no_impute(self):
        """Test no imputation."""
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute="none",
            seed=SEED,
        )
        # Check number of NaNs
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 69319622
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 19808408
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 9910198

    def test_zero_impute(self):
        """Test zero imputation."""
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute="zero",
            seed=SEED,
        )
        # Check no NaNs post imputation
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_mean_impute(self):
        """Test mean imputation."""
        dataset = PhysioNet2012(
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
        dataset = PhysioNet2012(
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

        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute=impute_with_zero,
            seed=SEED,
        )
        # Check number of NaNs
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 0

    def test_custom_imputation_2(self):
        """Test custom imputation function."""

        def no_imputation(X, y, fill, select):
            return X, y

        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute=no_imputation,
            seed=SEED,
        )
        # Check number of NaNs
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 69319622
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 19808408
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 9910198

    def test_time(self):
        """Test time argument."""
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([8400, 215, 43])
        assert dataset.X_val.shape == torch.Size([2400, 215, 43])
        assert dataset.X_test.shape == torch.Size([1200, 215, 43])
        # Check time channel
        for i in range(215):
            assert torch.equal(
                dataset.X_train[:, i, 0],
                torch.full([8400], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 0],
                torch.full([2400], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 0],
                torch.full([1200], fill_value=i, dtype=torch.float),
            )

    def test_no_time(self):
        """Test time argument."""
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([8400, 215, 42])
        assert dataset.X_val.shape == torch.Size([2400, 215, 42])
        assert dataset.X_test.shape == torch.Size([1200, 215, 42])

    def test_mask(self):
        """Test mask argument."""
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            mask=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([8400, 215, 84])
        assert dataset.X_val.shape == torch.Size([2400, 215, 84])
        assert dataset.X_test.shape == torch.Size([1200, 215, 84])

    def test_delta(self):
        """Test time delta argument."""
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            delta=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([8400, 215, 84])
        assert dataset.X_val.shape == torch.Size([2400, 215, 84])
        assert dataset.X_test.shape == torch.Size([1200, 215, 84])
        # Check time delta channel
        assert torch.equal(
            dataset.X_train[:, 0, 42], torch.zeros([8400], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_val[:, 0, 42], torch.zeros([2400], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_test[:, 0, 42], torch.zeros([1200], dtype=torch.float)
        )

    def test_time_mask_delta(self):
        """Test combination of time/mask/delta arguments."""
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            mask=True,
            delta=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([8400, 215, 127])
        assert dataset.X_val.shape == torch.Size([2400, 215, 127])
        assert dataset.X_test.shape == torch.Size([1200, 215, 127])
        # Check time channel
        for i in range(215):
            assert torch.equal(
                dataset.X_train[:, i, 0],
                torch.full([8400], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 0],
                torch.full([2400], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 0],
                torch.full([1200], fill_value=i, dtype=torch.float),
            )
        # Check time delta channel
        assert torch.equal(
            dataset.X_train[:, 0, 85], torch.zeros([8400], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_val[:, 0, 85], torch.zeros([2400], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_test[:, 0, 85], torch.zeros([1200], dtype=torch.float)
        )

    def test_standarisation(self):
        """Check training data is standardised."""
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            time=False,
            standardise=True,
            seed=SEED,
        )
        for c, Xc in enumerate(dataset.X_train.unbind(dim=-1)):
            print(Xc)
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
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check first value in 39th channel
        assert torch.allclose(
            dataset.X_train[0, 0, 39], torch.tensor(80.0), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 0, 39], torch.tensor(63.0), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 0, 39], torch.tensor(61.0), rtol=RTOL, atol=ATOL
        )

    def test_reproducibility_2(self):
        """Test seed argument."""
        dataset = PhysioNet2012(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=999999,
        )
        # Check first value in 39th channel
        assert torch.allclose(
            dataset.X_train[0, 0, 39], torch.tensor(49.0), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 0, 39], torch.tensor(64.0), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 0, 39], torch.tensor(47.0), rtol=RTOL, atol=ATOL
        )
