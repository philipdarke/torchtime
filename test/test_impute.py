import re

import pytest
import torch

from torchtime.data import UEA
from torchtime.impute import forward_impute, replace_missing


class TestReplace:
    def test_fill_type(self):
        """Test tensor with greater than 3 dimensions."""
        input_tensor = torch.ones((1, 2, 3))
        with pytest.raises(
            AssertionError, match=re.escape("argument 'fill' must be a Tensor")
        ):
            replace_missing(input_tensor, fill=[1, 2, 3])

    def test_fill_length_1(self):
        """Test length of ``fill`` argument."""
        input_tensor = torch.ones((1, 2, 3))
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "tensor 'fill' must have same number of channels as input (3)"
            ),
        ):
            replace_missing(input_tensor, fill=torch.tensor([1, 2]))

    def test_fill_length_2(self):
        """Test length of ``fill`` argument."""
        input_tensor = torch.ones((1, 2, 3))
        with pytest.raises(
            AssertionError,
            match=re.escape("'select' must be a list the same length as 'fill' (2)"),
        ):
            replace_missing(input_tensor, fill=torch.tensor([1, 2]), select=[0, 1, 2])

    def test_select_type(self):
        """Test type check of ``select`` argument."""
        input_tensor = torch.ones((1, 2, 3))
        with pytest.raises(
            AssertionError,
            match=re.escape("'select' must be a list the same length as 'fill' (2)"),
        ):
            replace_missing(input_tensor, fill=torch.tensor([1, 2]), select={0, 1, 2})

    def test_tensor(self):
        """Test with an example tensor."""
        input_tensor = torch.tensor(
            [
                [float("nan"), 2.0],
                [float("nan"), 4.0],
                [5.0, float("nan")],
                [float("nan"), 8.0],
                [9.0, float("nan")],
                [11.0, 12.0],
            ]
        )
        test_tensor = replace_missing(input_tensor, fill=torch.tensor([111.0, 222.0]))
        expect_tensor = torch.tensor(
            [
                [111.0, 2.0],
                [111.0, 4.0],
                [5.0, 222.0],
                [111.0, 8.0],
                [9.0, 222.0],
                [11.0, 12.0],
            ]
        )
        assert torch.equal(test_tensor, expect_tensor)

    def test_tensor_select(self):
        """Test select implementation with an example tensor."""
        input_tensor = torch.tensor(
            [
                [float("nan"), 2.0],
                [float("nan"), 4.0],
                [5.0, float("nan")],
                [float("nan"), 8.0],
                [9.0, float("nan")],
                [11.0, 12.0],
            ]
        )
        test_tensor = replace_missing(
            input_tensor, fill=torch.tensor([111.0]), select=[0]
        )
        expect_tensor = torch.tensor(
            [
                [111.0, 2.0],
                [111.0, 4.0],
                [5.0, float("nan")],
                [111.0, 8.0],
                [9.0, float("nan")],
                [11.0, 12.0],
            ]
        )
        assert torch.allclose(test_tensor, expect_tensor, equal_nan=True)

    def test_dataset(self):
        """Test with a dataset."""
        dataset = UEA(
            dataset="CharacterTrajectories",
            split="test",
            missing=[0.1, 0.5, 0.9],
            train_prop=0.7,
            val_prop=0.2,
            seed=456789,
        )
        # Check no NaNs post imputation
        X_impute = replace_missing(
            dataset.X, fill=torch.Tensor([1, 2, 3]), select=[1, 2, 3]
        )
        assert torch.sum(torch.isnan(X_impute)).item() == 0


class TestForward:
    def test_dim_check(self):
        """Test tensor with greater than 3 dimensions."""
        input_tensor = torch.ones((1, 2, 3, 4))
        with pytest.raises(
            AssertionError,
            match=re.escape("tensor 'input' must have 3 or fewer dimensions"),
        ):
            forward_impute(input_tensor)

    def test_no_fill_1(self):
        """Test check for ``fill`` argument (required)."""
        input_tensor = torch.tensor(
            [
                [float("nan"), 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ]
        )
        with pytest.raises(
            AssertionError, match=re.escape("argument 'fill' must be provided")
        ):
            forward_impute(input_tensor)

    def test_no_fill_2(self):
        """Test check for ``fill`` argument (not required)."""
        input_tensor = torch.tensor(
            [
                [1.0, 2.0],
                [float("nan"), 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ]
        )
        test_tensor = forward_impute(input_tensor)
        expect_tensor = torch.tensor(
            [
                [1.0, 2.0],
                [1.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ]
        )
        assert torch.equal(test_tensor, expect_tensor)

    def test_tensor(self):
        """Test with an example tensor."""
        input_tensor = torch.tensor(
            [
                [float("nan"), 2.0],
                [float("nan"), 4.0],
                [5.0, float("nan")],
                [float("nan"), 8.0],
                [9.0, float("nan")],
                [11.0, 12.0],
            ]
        )
        test_tensor = forward_impute(input_tensor, fill=torch.tensor([111.0, 222.0]))
        expect_tensor = torch.tensor(
            [
                [111.0, 2.0],
                [111.0, 4.0],
                [5.0, 4.0],
                [5.0, 8.0],
                [9.0, 8.0],
                [11.0, 12.0],
            ]
        )
        assert torch.equal(test_tensor, expect_tensor)

    def test_dataset(self):
        """Test with a dataset."""
        dataset = UEA(
            dataset="CharacterTrajectories",
            split="test",
            missing=[0.1, 0.5, 0.9],
            train_prop=0.7,
            val_prop=0.2,
            seed=456789,
        )
        # Check no NaNs post imputation
        X_impute = forward_impute(dataset.X, fill=torch.Tensor([float("nan"), 1, 2, 3]))
        assert torch.sum(torch.isnan(X_impute)).item() == 0
