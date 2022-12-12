import re

import pytest
import torch

from torchtime.constants import OBJ_EXT
from torchtime.data import PhysioNet2019
from torchtime.utils import _get_SHA256

SEED = 456789
RTOL = 1e-4
ATOL = 1e-4


class TestPhysioNet2019:
    """Test PhysioNet2019 class."""

    def test_invalid_split_arg(self):
        """Catch invalid split argument."""
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'split' must be one of ['train', 'val']"),
        ):
            PhysioNet2019(
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
            PhysioNet2019(
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
            PhysioNet2019(
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
            PhysioNet2019(
                split="test",
                train_prop=new_prop,
                val_prop=new_prop,
                seed=SEED,
            )

    def test_load_data(self):
        """Validate data set."""
        PhysioNet2019(
            split="train",
            train_prop=0.7,
            seed=SEED,
        )
        assert (
            _get_SHA256(".torchtime/physionet_2019/X" + OBJ_EXT)
            == "6eb80cddc4eb4fe1c4cdcae8ad35becc75469737542ca66ac417fda90f3a1db3"
        )
        assert (
            _get_SHA256(".torchtime/physionet_2019/y" + OBJ_EXT)
            == "8fa4a9f7f8fc532aca0a9e0df6c6fe837c3ae3d070ceb28054259ff97c8241a5"
        )
        assert (
            _get_SHA256(".torchtime/physionet_2019/length" + OBJ_EXT)
            == "829c06fb86444f2ca806371583cd38fe2d0e29b9045ae6a4cad306bd4f4fad1f"
        )

    def test_train_val(self):
        """Test training/validation split sizes."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28236, 336, 41])
        assert dataset.y_train.shape == torch.Size([28236, 336, 1])
        assert dataset.length_train.shape == torch.Size([28236])
        assert dataset.X_val.shape == torch.Size([12100, 336, 41])
        assert dataset.y_val.shape == torch.Size([12100, 336, 1])
        assert dataset.length_val.shape == torch.Size([12100])
        # Ensure no test data is returned
        with pytest.raises(
            AttributeError,
            match=re.escape("'PhysioNet2019' object has no attribute 'X_test'"),
        ):
            dataset.X_test
        with pytest.raises(
            AttributeError,
            match=re.escape("'PhysioNet2019' object has no attribute 'y_test'"),
        ):
            dataset.y_test
        with pytest.raises(
            AttributeError,
            match=re.escape("'PhysioNet2019' object has no attribute 'length_test'"),
        ):
            dataset.length_test

    def test_train_val_test(self):
        """Test training/validation/test split sizes."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28236, 336, 41])
        assert dataset.y_train.shape == torch.Size([28236, 336, 1])
        assert dataset.length_train.shape == torch.Size([28236])
        assert dataset.X_val.shape == torch.Size([8067, 336, 41])
        assert dataset.y_val.shape == torch.Size([8067, 336, 1])
        assert dataset.length_val.shape == torch.Size([8067])
        assert dataset.X_test.shape == torch.Size([4033, 336, 41])
        assert dataset.y_test.shape == torch.Size([4033, 336, 1])
        assert dataset.length_test.shape == torch.Size([4033])

    def test_train_split(self):
        """Test training split is returned."""
        dataset = PhysioNet2019(
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
        dataset = PhysioNet2019(
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
        dataset = PhysioNet2019(
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
        dataset = PhysioNet2019(
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
            PhysioNet2019(
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
            PhysioNet2019(
                split="train",
                train_prop=0.7,
                impute=3,
                seed=SEED,
            )

    def test_no_impute(self):
        """Test no imputation."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute="none",
            seed=SEED,
        )
        # Check number of NaNs
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 366491058
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 8401181
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 104715681
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 2399984
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 52332856
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 1199521

    def test_zero_impute(self):
        """Test zero imputation."""
        dataset = PhysioNet2019(
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
        dataset = PhysioNet2019(
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
        dataset = PhysioNet2019(
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

        dataset = PhysioNet2019(
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
            """Does not impute data i.e. same as impute='none'"""
            return X, y

        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute=no_imputation,
            seed=SEED,
        )
        # Check number of NaNs
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 366491058
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 8401181
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 104715681
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 2399984
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 52332856
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 1199521

    def test_overwrite_data(self):
        """Overwrite cache and validate data set."""
        PhysioNet2019(
            split="train",
            train_prop=0.7,
            seed=SEED,
            overwrite_cache=True,
        )
        assert (
            _get_SHA256(".torchtime/physionet_2019/X" + OBJ_EXT)
            == "ea94129dd03e594da80efae132b674b284499c613174d0314e13e8aaac179ba7"
        )
        assert (
            _get_SHA256(".torchtime/physionet_2019/y" + OBJ_EXT)
            == "5f3cf8f30e4bebf166c5a4f9d4c2030dfb82d57729e6b4e67409047150345c0e"
        )
        assert (
            _get_SHA256(".torchtime/physionet_2019/length" + OBJ_EXT)
            == "209672aa41dc2f092a63e49accc4203598b2d4deb9b28b910cbbb4082518c618"
        )

    def test_time(self):
        """Test time argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28236, 336, 41])
        assert dataset.X_val.shape == torch.Size([8067, 336, 41])
        assert dataset.X_test.shape == torch.Size([4033, 336, 41])
        # Check time channel
        for i in range(182):
            assert torch.equal(
                dataset.X_train[:, i, 0],
                torch.full([28236], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 0],
                torch.full([8067], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 0],
                torch.full([4033], fill_value=i, dtype=torch.float),
            )

    def test_no_time(self):
        """Test time argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28236, 336, 40])
        assert dataset.X_val.shape == torch.Size([8067, 336, 40])
        assert dataset.X_test.shape == torch.Size([4033, 336, 40])

    def test_mask(self):
        """Test mask argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            mask=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28236, 336, 80])
        assert dataset.X_val.shape == torch.Size([8067, 336, 80])
        assert dataset.X_test.shape == torch.Size([4033, 336, 80])

    def test_delta(self):
        """Test time delta argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            delta=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28236, 336, 80])
        assert dataset.X_val.shape == torch.Size([8067, 336, 80])
        assert dataset.X_test.shape == torch.Size([4033, 336, 80])
        # Check time delta channel
        assert torch.equal(
            dataset.X_train[:, 0, 40], torch.zeros([28236], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_val[:, 0, 40], torch.zeros([8067], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_test[:, 0, 40], torch.zeros([4033], dtype=torch.float)
        )

    def test_time_mask_delta(self):
        """Test combination of time/mask/delta arguments."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            mask=True,
            delta=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([28236, 336, 121])
        assert dataset.X_val.shape == torch.Size([8067, 336, 121])
        assert dataset.X_test.shape == torch.Size([4033, 336, 121])
        # Check time channel
        for i in range(182):
            assert torch.equal(
                dataset.X_train[:, i, 0],
                torch.full([28236], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 0],
                torch.full([8067], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 0],
                torch.full([4033], fill_value=i, dtype=torch.float),
            )
        # Check time delta channel
        assert torch.equal(
            dataset.X_train[:, 0, 81], torch.zeros([28236], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_val[:, 0, 81], torch.zeros([8067], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_test[:, 0, 81], torch.zeros([4033], dtype=torch.float)
        )

    def test_standarisation(self):
        """Check training data is standardised."""
        dataset = PhysioNet2019(
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
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check first value in 39th channel
        assert torch.allclose(
            dataset.X_train[0, 0, 39], torch.tensor(-1.53), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 0, 39], torch.tensor(-0.02), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 0, 39], torch.tensor(-6.73), rtol=RTOL, atol=ATOL
        )

    def test_reproducibility_2(self):
        """Test seed argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=999999,
        )
        # Check first value in 39th channel
        assert torch.allclose(
            dataset.X_train[0, 0, 39], torch.tensor(-13.01), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 0, 39], torch.tensor(-109.75), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 0, 39], torch.tensor(-131.18), rtol=RTOL, atol=ATOL
        )
