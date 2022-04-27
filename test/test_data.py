import re

import pytest
import torch

from torchtime.data import TensorTimeSeriesDataset

SEED = 456789


class TestTensorTimeSeriesDataset:
    """Test size validation."""

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
        """Test simple data set."""
        X = torch.ones([1234, 100, 10])
        y = torch.ones([1234, 1])
        length = torch.ones([1234])
        dataset = TensorTimeSeriesDataset(
            dataset="test_tensordataset_simple",
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

    def test_imputation_validation_1(self):
        X = torch.tensor(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 6],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 12],
                    [4, 5, 12],
                    [7, 8, 12],
                    [float("nan"), float("nan"), float("nan")],
                ],
            ]
        )
        y = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
        length = torch.tensor([4, 4, 4, 4, 3])
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'categorical' must be a list"),
        ):
            TensorTimeSeriesDataset(
                dataset="test_tensordataset_imputation_validation",
                X=X,
                y=y,
                length=length,
                split="train",
                train_prop=0.7,
                impute="mean",
                categorical=3,
                seed=SEED,
            )
        with pytest.raises(
            AssertionError,
            match=re.escape("indices in 'categorical' should be between 0 and 2"),
        ):
            TensorTimeSeriesDataset(
                dataset="test_tensordataset_imputation_validation",
                X=X,
                y=y,
                length=length,
                split="train",
                train_prop=0.7,
                impute="mean",
                categorical=[3],
                seed=SEED,
            )

    def test_imputation_validation_2(self):
        X = torch.tensor(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 6],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 12],
                    [4, 5, 12],
                    [7, 8, 12],
                    [float("nan"), float("nan"), float("nan")],
                ],
            ]
        )
        y = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
        length = torch.tensor([4, 4, 4, 4, 3])
        with pytest.raises(
            AssertionError,
            match=re.escape("argument 'override' must be a dictionary"),
        ):
            TensorTimeSeriesDataset(
                dataset="test_tensordataset_imputation_validation",
                X=X,
                y=y,
                length=length,
                split="train",
                train_prop=0.7,
                impute="mean",
                override=3,
                seed=SEED,
            )
        with pytest.raises(
            AssertionError,
            match=re.escape("indices in 'override' should be between 0 and 2"),
        ):
            TensorTimeSeriesDataset(
                dataset="test_tensordataset_imputation_validation",
                X=X,
                y=y,
                length=length,
                split="train",
                train_prop=0.7,
                impute="mean",
                override={3: 10},
                seed=SEED,
            )

    def test_imputation_1(self):
        """Test imputation with categorical variables."""
        X = torch.tensor(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 6],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 12],
                    [4, 5, 12],
                    [7, 8, 12],
                    [float("nan"), float("nan"), float("nan")],
                ],
            ]
        )
        y = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
        length = torch.tensor([4, 4, 4, 4, 3])
        dataset = TensorTimeSeriesDataset(
            dataset="test_tensordataset_imputation_1",
            X=X,
            y=y,
            length=length,
            split="train",
            train_prop=0.7,
            impute="mean",
            categorical=[2],
            time=False,
            seed=SEED,
        )
        assert torch.allclose(dataset.X_train[0, 3], torch.tensor([5.2, 6.2, 12.0]))

    def test_imputation_2(self):
        """Test imputation with categorical variables and overridden means."""
        X = torch.tensor(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 6],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                ],
                [
                    [1, 2, 12],
                    [4, 5, 12],
                    [7, 8, 12],
                    [float("nan"), float("nan"), float("nan")],
                ],
            ]
        )
        y = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
        length = torch.tensor([4, 4, 4, 4, 3])
        dataset = TensorTimeSeriesDataset(
            dataset="test_tensordataset_imputation_2",
            X=X,
            y=y,
            length=length,
            split="train",
            train_prop=0.7,
            impute="mean",
            categorical=[2],
            override={1: 10},
            time=False,
            seed=SEED,
        )
        assert torch.allclose(dataset.X_train[0, 3], torch.tensor([5.2, 10.0, 12.0]))
