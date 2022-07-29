import re

import pytest
import torch

from torchtime.constants import OBJ_EXT
from torchtime.data import UEA
from torchtime.utils import _get_SHA256

DATASET = "ArrowHead"
SEED = 456789
RTOL = 1e-4
ATOL = 1e-4


class TestUEAArrowHead:
    """Test UEA class with ArrowHead data set."""

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
        assert (
            _get_SHA256(".torchtime/uea_" + DATASET + "/X" + OBJ_EXT)
            == "9530efe27c6450c5da88a44c72ecd584b80c82599d62a3a02e0e09f572eb3a38"
        )
        assert (
            _get_SHA256(".torchtime/uea_" + DATASET + "/y" + OBJ_EXT)
            == "7f08d6239b17cad032fdc9a2d1f607b500825167a9884e2ad84f423a7513a30c"
        )
        assert (
            _get_SHA256(".torchtime/uea_" + DATASET + "/length" + OBJ_EXT)
            == "7348daeb7eb5239a1e400df18c574daabb03764e6c0422590c2ed44b014f9160"
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
        assert dataset.X_train.shape == torch.Size([148, 251, 2])
        assert dataset.y_train.shape == torch.Size([148, 3])
        assert dataset.length_train.shape == torch.Size([148])
        assert dataset.X_val.shape == torch.Size([63, 251, 2])
        assert dataset.y_val.shape == torch.Size([63, 3])
        assert dataset.length_val.shape == torch.Size([63])
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
        assert dataset.X_train.shape == torch.Size([148, 251, 2])
        assert dataset.y_train.shape == torch.Size([148, 3])
        assert dataset.length_train.shape == torch.Size([148])
        assert dataset.X_val.shape == torch.Size([42, 251, 2])
        assert dataset.y_val.shape == torch.Size([42, 3])
        assert dataset.length_val.shape == torch.Size([42])
        assert dataset.X_test.shape == torch.Size([21, 251, 2])
        assert dataset.y_test.shape == torch.Size([21, 3])
        assert dataset.length_test.shape == torch.Size([21])

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
        assert torch.allclose(dataset.X, dataset.X_train)
        assert torch.allclose(dataset.y, dataset.y_train)
        assert torch.allclose(dataset.length, dataset.length_train)

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
        assert torch.allclose(dataset.X, dataset.X_val)
        assert torch.allclose(dataset.y, dataset.y_val)
        assert torch.allclose(dataset.length, dataset.length_val)

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
        assert torch.allclose(dataset.X, dataset.X_test)
        assert torch.allclose(dataset.y, dataset.y_test)
        assert torch.allclose(dataset.length, dataset.length_test)

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
        assert torch.all(dataset.length_train == dataset.length_train[0])
        assert dataset.X_train.size(1) == dataset.length_train[0]
        assert torch.all(dataset.length_val == dataset.length_val[0])
        assert dataset.X_val.size(1) == dataset.length_val[0]
        assert torch.all(dataset.length_test == dataset.length_test[0])
        assert dataset.X_test.size(1) == dataset.length_test[0]

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
            torch.sum(torch.isnan(dataset.X_train)).item() == 18500
        )  # expect around 148 * 251 * 0.5 = 18,574
        assert (
            torch.sum(torch.isnan(dataset.X_val)).item() == 5250
        )  # expect around 42 * 251 * 0.5 = 5,271
        assert (
            torch.sum(torch.isnan(dataset.X_test)).item() == 2625
        )  # expect around 21 * 251 * 0.5 = 2,535

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
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 18500
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 5250
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 2625
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
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 0
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
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 0
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
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 18500
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 5250
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 2625
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

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
        assert dataset.X_train.shape == torch.Size([148, 251, 2])
        assert dataset.X_val.shape == torch.Size([42, 251, 2])
        assert dataset.X_test.shape == torch.Size([21, 251, 2])
        # Check time channel
        for i in range(251):
            assert torch.equal(
                dataset.X_train[:, i, 0],
                torch.full([148], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 0],
                torch.full([42], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 0],
                torch.full([21], fill_value=i, dtype=torch.float),
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
        assert dataset.X_train.shape == torch.Size([148, 251, 1])
        assert dataset.X_val.shape == torch.Size([42, 251, 1])
        assert dataset.X_test.shape == torch.Size([21, 251, 1])

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
        assert dataset.X_train.shape == torch.Size([148, 251, 2])
        assert dataset.X_val.shape == torch.Size([42, 251, 2])
        assert dataset.X_test.shape == torch.Size([21, 251, 2])
        # Check mask channel
        assert torch.sum(dataset.X_train[:, :, 1]) == 148 * 251
        assert torch.sum(dataset.X_val[:, :, 1]) == 42 * 251
        assert torch.sum(dataset.X_test[:, :, 1]) == 21 * 251

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
        assert dataset.X_train.shape == torch.Size([148, 251, 2])
        assert dataset.X_val.shape == torch.Size([42, 251, 2])
        assert dataset.X_test.shape == torch.Size([21, 251, 2])
        # Check time delta channel
        assert torch.equal(
            dataset.X_train[:, 0, 1], torch.zeros([148], dtype=torch.float)
        )
        assert torch.equal(dataset.X_val[:, 0, 1], torch.zeros([42], dtype=torch.float))
        assert torch.equal(
            dataset.X_test[:, 0, 1], torch.zeros([21], dtype=torch.float)
        )
        for i in range(1, 251):
            assert torch.equal(
                dataset.X_train[:, i, 1],
                torch.full([148], fill_value=1, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 1],
                torch.full([42], fill_value=1, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 1],
                torch.full([21], fill_value=1, dtype=torch.float),
            )

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
        assert dataset.X_train.shape == torch.Size([148, 251, 4])
        assert dataset.X_val.shape == torch.Size([42, 251, 4])
        assert dataset.X_test.shape == torch.Size([21, 251, 4])
        # Check time channel
        for i in range(251):
            assert torch.equal(
                dataset.X_train[:, i, 0],
                torch.full([148], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 0],
                torch.full([42], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 0],
                torch.full([21], fill_value=i, dtype=torch.float),
            )
        # Check mask channel
        assert torch.sum(dataset.X_train[:, :, 2]) == 148 * 251
        assert torch.sum(dataset.X_val[:, :, 2]) == 42 * 251
        assert torch.sum(dataset.X_test[:, :, 2]) == 21 * 251
        # Check time delta channel
        assert torch.equal(
            dataset.X_train[:, 0, 3], torch.zeros([148], dtype=torch.float)
        )
        assert torch.equal(dataset.X_val[:, 0, 3], torch.zeros([42], dtype=torch.float))
        assert torch.equal(
            dataset.X_test[:, 0, 3], torch.zeros([21], dtype=torch.float)
        )
        for i in range(1, 251):
            assert torch.equal(
                dataset.X_train[:, i, 3],
                torch.full([148], fill_value=1, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 3],
                torch.full([42], fill_value=1, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 3],
                torch.full([21], fill_value=1, dtype=torch.float),
            )

    def test_standarisation(self):
        """Check training data is standardised."""
        dataset = UEA(
            dataset=DATASET,
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
        dataset = UEA(
            dataset=DATASET,
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check first value in each data set
        assert torch.allclose(
            dataset.X_train[0, 0, 1], torch.tensor(-1.8515), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 0, 1], torch.tensor(-1.9190), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 0, 1], torch.tensor(-1.8091), rtol=RTOL, atol=ATOL
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
            dataset.X_train[0, 0, 1], torch.tensor(-1.7993), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 0, 1], torch.tensor(-2.1308), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 0, 1], torch.tensor(-2.1468), rtol=RTOL, atol=ATOL
        )
