import re

import pytest
import torch

from torchtime.data import PhysioNet2019

DOWNSCALE = 0.01
SEED = 456789
RTOL = 1e-4
ATOL = 1e-4


class TestPhysioNet2019:
    def test_invalid_split_arg(self):
        """Catch invalid split argument."""
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'split' must be one of ['train', 'val']"),
        ):
            PhysioNet2019(
                split="xyz",
                train_prop=0.8,
                downscale=DOWNSCALE,
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
                downscale=DOWNSCALE,
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
                downscale=DOWNSCALE,
                seed=SEED,
            )
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'val_prop' must be in range (0, 1-train_prop)"),
        ):
            PhysioNet2019(
                split="test",
                train_prop=0.5,
                val_prop=0.5,
                downscale=DOWNSCALE,
                seed=SEED,
            )

    def test_train_val(self):
        """Test training/validation split sizes."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            downscale=DOWNSCALE,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([283, 190, 41])
        assert dataset.y_train.shape == torch.Size([283, 190, 1])
        assert dataset.length_train.shape == torch.Size([283])
        assert dataset.X_val.shape == torch.Size([120, 190, 41])
        assert dataset.y_val.shape == torch.Size([120, 190, 1])
        assert dataset.length_val.shape == torch.Size([120])
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
            downscale=DOWNSCALE,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([283, 190, 41])
        assert dataset.y_train.shape == torch.Size([283, 190, 1])
        assert dataset.length_train.shape == torch.Size([283])
        assert dataset.X_val.shape == torch.Size([80, 190, 41])
        assert dataset.y_val.shape == torch.Size([80, 190, 1])
        assert dataset.length_val.shape == torch.Size([80])
        assert dataset.X_test.shape == torch.Size([40, 190, 41])
        assert dataset.y_test.shape == torch.Size([40, 190, 1])
        assert dataset.length_test.shape == torch.Size([40])

    def test_train_split(self):
        """Test training split is returned."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            downscale=DOWNSCALE,
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
            downscale=DOWNSCALE,
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
            downscale=DOWNSCALE,
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
            downscale=DOWNSCALE,
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
                "argument 'impute' must be a string in dict_keys(['none', 'mean', 'forward']) or a function"  # noqa: E501
            ),
        ):
            PhysioNet2019(
                split="train",
                train_prop=0.7,
                impute="blah",
                downscale=DOWNSCALE,
                seed=SEED,
            )
        with pytest.raises(
            Exception,
            match=re.escape(
                "argument 'impute' must be a string in dict_keys(['none', 'mean', 'forward']) or a function"  # noqa: E501
            ),
        ):
            PhysioNet2019(
                split="train",
                train_prop=0.7,
                impute=3,
                downscale=DOWNSCALE,
                seed=SEED,
            )

    def test_no_impute(self):
        """Test no imputation."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute="none",
            downscale=DOWNSCALE,
            seed=SEED,
        )
        # Check number of NaNs
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 2023739
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 43146
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 570332
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 12045
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 285244
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 6037

    def test_mean_impute(self):
        """Test mean imputation."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute="mean",
            downscale=DOWNSCALE,
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
            downscale=DOWNSCALE,
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

        def custom_imputer(X, y, fill):
            """Does not impute data i.e. same as impute='none'"""
            return X, y

        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            impute=custom_imputer,
            downscale=DOWNSCALE,
            seed=SEED,
        )
        # Check number of NaNs
        assert torch.sum(torch.isnan(dataset.X_train)).item() == 2023739
        assert torch.sum(torch.isnan(dataset.y_train)).item() == 43146
        assert torch.sum(torch.isnan(dataset.X_val)).item() == 570332
        assert torch.sum(torch.isnan(dataset.y_val)).item() == 12045
        assert torch.sum(torch.isnan(dataset.X_test)).item() == 285244
        assert torch.sum(torch.isnan(dataset.y_test)).item() == 6037

    def test_time(self):
        """Test time argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=True,
            downscale=DOWNSCALE,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([283, 190, 41])
        assert dataset.X_val.shape == torch.Size([80, 190, 41])
        assert dataset.X_test.shape == torch.Size([40, 190, 41])
        # Check time channel
        for i in range(182):
            assert torch.equal(
                dataset.X_train[:, i, 0],
                torch.full([283], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 0],
                torch.full([80], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 0],
                torch.full([40], fill_value=i, dtype=torch.float),
            )

    def test_no_time(self):
        """Test time argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            downscale=DOWNSCALE,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([283, 190, 40])
        assert dataset.X_val.shape == torch.Size([80, 190, 40])
        assert dataset.X_test.shape == torch.Size([40, 190, 40])

    def test_mask(self):
        """Test mask argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            mask=True,
            downscale=DOWNSCALE,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([283, 190, 80])
        assert dataset.X_val.shape == torch.Size([80, 190, 80])
        assert dataset.X_test.shape == torch.Size([40, 190, 80])

    def test_delta(self):
        """Test time delta argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            time=False,
            delta=True,
            downscale=DOWNSCALE,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([283, 190, 80])
        assert dataset.X_val.shape == torch.Size([80, 190, 80])
        assert dataset.X_test.shape == torch.Size([40, 190, 80])
        # Check time delta channel
        assert torch.equal(
            dataset.X_train[:, 0, 40], torch.zeros([283], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_val[:, 0, 40], torch.zeros([80], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_test[:, 0, 40], torch.zeros([40], dtype=torch.float)
        )

    def test_time_mask_delta(self):
        """Test combination of time/mask/delta arguments."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            mask=True,
            delta=True,
            downscale=DOWNSCALE,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([283, 190, 121])
        assert dataset.X_val.shape == torch.Size([80, 190, 121])
        assert dataset.X_test.shape == torch.Size([40, 190, 121])
        # Check time channel
        for i in range(182):
            assert torch.equal(
                dataset.X_train[:, i, 0],
                torch.full([283], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_val[:, i, 0],
                torch.full([80], fill_value=i, dtype=torch.float),
            )
            assert torch.equal(
                dataset.X_test[:, i, 0],
                torch.full([40], fill_value=i, dtype=torch.float),
            )
        # Check time delta channel
        assert torch.equal(
            dataset.X_train[:, 0, 81], torch.zeros([283], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_val[:, 0, 81], torch.zeros([80], dtype=torch.float)
        )
        assert torch.equal(
            dataset.X_test[:, 0, 81], torch.zeros([40], dtype=torch.float)
        )

    def test_downscale(self):
        """Test downscale argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            downscale=DOWNSCALE * 2,
            seed=SEED,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([565, 190, 41])
        assert dataset.y_train.shape == torch.Size([565, 190, 1])
        assert dataset.length_train.shape == torch.Size([565])
        assert dataset.X_val.shape == torch.Size([161, 190, 41])
        assert dataset.y_val.shape == torch.Size([161, 190, 1])
        assert dataset.length_val.shape == torch.Size([161])
        assert dataset.X_test.shape == torch.Size([80, 190, 41])
        assert dataset.y_test.shape == torch.Size([80, 190, 1])
        assert dataset.length_test.shape == torch.Size([80])

    def test_reproducibility_1(self):
        """Test seed argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            downscale=DOWNSCALE,
            seed=SEED,
        )
        # Check first value in each data set
        assert torch.allclose(
            dataset.X_train[0, 0, 1], torch.tensor(85.5), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 1, 1], torch.tensor(92.0), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 2, 1], torch.tensor(83.0), rtol=RTOL, atol=ATOL
        )

    def test_reproducibility_2(self):
        """Test seed argument."""
        dataset = PhysioNet2019(
            split="train",
            train_prop=0.7,
            val_prop=0.2,
            downscale=DOWNSCALE,
            seed=999999,
        )
        # Check first value in each data set
        assert torch.allclose(
            dataset.X_train[0, 0, 1], torch.tensor(92.0), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_val[0, 0, 1], torch.tensor(98.0), rtol=RTOL, atol=ATOL
        )
        assert torch.allclose(
            dataset.X_test[0, 1, 1], torch.tensor(72.5), rtol=RTOL, atol=ATOL
        )
