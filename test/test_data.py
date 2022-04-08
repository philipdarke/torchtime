import re

import pytest
import torch

from torchtime.data import UEA, TensorTimeSeriesDataset

SEED = 456789
RTOL = 1e-4
ATOL = 1e-4

# class TestPhysioNet2019:
#     """GITHUB ACTIONS RUNNER HAS INSUFFICIENT MEMORY FOR THIS TEST"""
#
#     def test_dataset(self):
#         dataset = PhysioNet2019(split="test", train_prop=0.7, val_prop=0.2, seed=SEED)
#         # Check data set size
#         assert dataset.X_train.shape == torch.Size([28236, 336, 41])
#         assert dataset.y_train.shape == torch.Size([28236, 336, 1])
#         assert dataset.length_train.shape == torch.Size([28236])
#         assert dataset.X_val.shape == torch.Size([8067, 336, 41])
#         assert dataset.y_val.shape == torch.Size([8067, 336, 1])
#         assert dataset.length_val.shape == torch.Size([8067])
#         assert dataset.X_test.shape == torch.Size([4033, 336, 41])
#         assert dataset.y_test.shape == torch.Size([4033, 336, 1])
#         assert dataset.length_test.shape == torch.Size([4033])
#         # Check correct split is returned
#         assert torch.allclose(dataset.X, dataset.X_test, equal_nan=True)
#         assert torch.allclose(dataset.y, dataset.y_test, equal_nan=True)
#         assert torch.allclose(dataset.length, dataset.length_test, equal_nan=True)
#
#     def test_dataset_mask_delta(self):
#         dataset = PhysioNet2019(
#             split="test",
#             train_prop=0.7,
#             val_prop=0.2,
#             time=False,
#             mask=True,
#             delta=True,
#             seed=SEED,
#         )
#         # Check data set size
#         assert dataset.X_train.shape == torch.Size([28236, 336, 120])
#         assert dataset.X_val.shape == torch.Size([8067, 336, 120])
#         assert dataset.X_test.shape == torch.Size([4033, 336, 120])
#         # Check correct split is returned
#         assert torch.allclose(dataset.X, dataset.X_test, equal_nan=True)
#         assert torch.allclose(dataset.y, dataset.y_test, equal_nan=True)
#         assert torch.allclose(dataset.length, dataset.length_test, equal_nan=True)


class TestUEAArrowHead:
    def test_invalid_split_arg(self):
        """Catch invalid split argument."""
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'split' must be one of ['train', 'val']"),
        ):
            UEA(
                dataset="ArrowHead",
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
                dataset="ArrowHead",
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
                dataset="ArrowHead",
                split="train",
                train_prop=1,
                seed=SEED,
            )
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'val_prop' must be in range (0, 1-train_prop)"),
        ):
            UEA(
                dataset="ArrowHead",
                split="test",
                train_prop=0.5,
                val_prop=0.5,
                seed=SEED,
            )

    def test_train_val(self):
        """Test training/validation split sizes."""
        dataset = UEA(
            dataset="ArrowHead",
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
            dataset="ArrowHead",
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
            dataset="ArrowHead",
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
            dataset="ArrowHead",
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
            dataset="ArrowHead",
            split="test",
            train_prop=0.7,
            val_prop=0.2,
            seed=SEED,
        )
        # Check correct split is returned
        assert torch.allclose(dataset.X, dataset.X_test)
        assert torch.allclose(dataset.y, dataset.y_test)
        assert torch.allclose(dataset.length, dataset.length_test)

    def test_missing(self):
        """Test missing data simulation."""
        dataset = UEA(
            dataset="ArrowHead",
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
                "argument 'impute' must be a string in dict_keys(['none', 'mean', 'forward']) or a function"  # noqa: E501
            ),
        ):
            UEA(
                dataset="ArrowHead",
                split="train",
                train_prop=0.7,
                missing=0.5,
                impute="blah",
                seed=SEED,
            )
        with pytest.raises(
            Exception,
            match=re.escape(
                "argument 'impute' must be a string in dict_keys(['none', 'mean', 'forward']) or a function"  # noqa: E501
            ),
        ):
            UEA(
                dataset="ArrowHead",
                split="train",
                train_prop=0.7,
                missing=0.5,
                impute=3,
                seed=SEED,
            )

    def test_no_impute(self):
        """Test no imputation."""
        dataset = UEA(
            dataset="ArrowHead",
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

    def test_mean_impute(self):
        """Test mean imputation."""
        dataset = UEA(
            dataset="ArrowHead",
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
            dataset="ArrowHead",
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

    def test_custom_impute(self):
        """Test custom imputation function."""

        def custom_imputer(X, y):
            """Does not impute data i.e. same as impute='none'"""
            return X, y

        dataset = UEA(
            dataset="ArrowHead",
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            missing=0.5,
            impute=custom_imputer,
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
            dataset="ArrowHead",
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
            dataset="ArrowHead",
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
            dataset="ArrowHead",
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
            dataset="ArrowHead",
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
            dataset="ArrowHead",
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

    def test_downscale(self):
        """Test downscale argument."""
        dataset = UEA(
            dataset="ArrowHead",
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            downscale=0.1,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([15, 251, 2])
        assert dataset.y_train.shape == torch.Size([15, 3])
        assert dataset.length_train.shape == torch.Size([15])
        assert dataset.X_val.shape == torch.Size([4, 251, 2])
        assert dataset.y_val.shape == torch.Size([4, 3])
        assert dataset.length_val.shape == torch.Size([4])
        assert dataset.X_test.shape == torch.Size([2, 251, 2])
        assert dataset.y_test.shape == torch.Size([2, 3])
        assert dataset.length_test.shape == torch.Size([2])

    def test_reproducibility_1(self):
        """Test seed argument."""
        dataset = UEA(
            dataset="ArrowHead",
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
            dataset="ArrowHead",
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


class TestUEACharacterTrajectories:
    def test_chartraj_missing_by_channel_mask(self):
        dataset = UEA(
            dataset="CharacterTrajectories",
            split="val",
            missing=[0.1, 0.5, 0.9],
            train_prop=0.7,
            val_prop=0.2,
            mask=True,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([2002, 182, 7])
        assert dataset.X_val.shape == torch.Size([571, 182, 7])
        assert dataset.X_test.shape == torch.Size([285, 182, 7])
        # Check correct split is returned
        assert torch.allclose(dataset.X, dataset.X_val, equal_nan=True)
        assert torch.allclose(dataset.y, dataset.y_val, equal_nan=True)
        assert torch.allclose(dataset.length, dataset.length_val, equal_nan=True)
        # Check missing data
        assert (
            torch.sum(torch.isnan(dataset.X_train)).item() == 731772
        )  # expect around (2002 * 182 - 239917) * 3 + 239917 * 1.5 = 733,217
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 0
        assert (
            torch.sum(torch.isnan(dataset.X_val)).item() == 207990
        )  # expect around (571 * 182 - 68934) * 3 + 68934 * 1.5 = 208,365
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 0
        assert (
            torch.sum(torch.isnan(dataset.X_test)).item() == 104060
        )  # expect around (285 * 182 - 34088) * 3 + 34088 * 1.5 = 104,478
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 0

    def test_mean_impute(self):
        dataset = UEA(
            dataset="CharacterTrajectories",
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            missing=[0.1, 0.5, 0.9],
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


class TestTensorTimeSeriesDataset:
    def test_tensor_sizes(self):
        X = torch.ones([1234, 100, 10])
        y = torch.ones([123, 1])
        length = torch.ones([12])
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "arguments 'X', 'y' and 'length' must be the same size in dimension 0 (currently 1234/123/12 respectively)"  # noqa: E501
            ),
        ):
            TensorTimeSeriesDataset(
                dataset="test_tensordataset",
                X=X,
                y=y,
                length=length,
                split="train",
                train_prop=0.7,
                seed=SEED,
            )

    def test_simple(self):
        X = torch.ones([1234, 100, 10])
        y = torch.ones([1234, 1])
        length = torch.ones([1234])
        dataset = TensorTimeSeriesDataset(
            dataset="test_tensordataset",
            X=X,
            y=y,
            length=length,
            split="train",
            train_prop=0.7,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([864, 100, 11])
        assert dataset.y_train.shape == torch.Size([864, 1])
        assert dataset.length_train.shape == torch.Size([864])
        assert dataset.X_val.shape == torch.Size([370, 100, 11])
        assert dataset.y_val.shape == torch.Size([370, 1])
        assert dataset.length_val.shape == torch.Size([370])
