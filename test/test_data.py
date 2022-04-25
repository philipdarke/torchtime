import re

import pytest
import torch

from torchtime.data import TensorTimeSeriesDataset

SEED = 456789


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
