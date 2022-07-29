import re

import pytest
import torch

from torchtime.data import UEA
from torchtime.impute import forward_impute, replace_missing

SEED = 456789
RTOL = 1e-4
ATOL = 1e-4


@pytest.fixture
def tensor_missing():
    """Tensor with missing data for testing."""
    tensor = torch.tensor(
        [
            [float("nan"), 2.0],
            [float("nan"), 4.0],
            [5.0, float("nan")],
            [float("nan"), 8.0],
            [9.0, float("nan")],
            [11.0, 12.0],
        ]
    )
    yield tensor


class TestReplace:
    """Test replace_missing() imputation function."""

    def test_fill_type(self):
        """Test invalid fill input."""
        input_tensor = torch.ones((1, 2, 3))
        with pytest.raises(
            AssertionError, match=re.escape("argument 'fill' must be a Tensor")
        ):
            replace_missing(input_tensor, fill=[1, 2, 3])

    def test_fill_length(self):
        """Test length of ``fill`` argument."""
        # Without select argument
        input_tensor = torch.ones((1, 2, 3))
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "Tensor 'fill' must have same number of channels as input (3)"
            ),
        ):
            replace_missing(input_tensor, fill=torch.tensor([1, 2]))
        # With select argument
        input_tensor = torch.ones((1, 2, 3))
        with pytest.raises(
            AssertionError,
            match=re.escape("'select' must be a Tensor the same length as 'fill' (2)"),
        ):
            replace_missing(
                input_tensor, fill=torch.tensor([1, 2]), select=torch.tensor([0, 1, 2])
            )

    def test_select_type(self):
        """Test type check of ``select`` argument."""
        input_tensor = torch.ones((1, 2, 3))
        with pytest.raises(
            AssertionError,
            match=re.escape("'select' must be a Tensor the same length as 'fill' (2)"),
        ):
            replace_missing(input_tensor, fill=torch.tensor([1, 2]), select={0, 1, 2})

    def test_select_length(self):
        """Test length of ``select`` argument."""
        input_tensor = torch.ones((1, 2, 3))
        with pytest.raises(
            AssertionError,
            match=re.escape("'select' must be a Tensor the same length as 'fill' (2)"),
        ):
            replace_missing(
                input_tensor, fill=torch.tensor([1, 2]), select=torch.tensor([0, 1, 2])
            )

    def test_tensor(self, tensor_missing):
        """Test tensor."""
        test_tensor = replace_missing(tensor_missing, fill=torch.tensor([111.0, 222.0]))
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

    def test_tensor_select(self, tensor_missing):
        """Test ``select`` implementation."""
        test_tensor = replace_missing(
            tensor_missing, fill=torch.tensor([111.0]), select=torch.tensor([0])
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
            dataset.X, fill=torch.Tensor([1, 2, 3]), select=torch.Tensor([1, 2, 3])
        )
        assert torch.sum(torch.isnan(X_impute)).item() == 0


class TestForward:
    """Test forward_impute() imputation function."""

    def test_dim_check(self):
        """Test tensor with 1 dimension."""
        input_tensor = torch.ones((1))
        with pytest.raises(
            AssertionError,
            match=re.escape("Tensor 'input' must have at least two dimensions"),
        ):
            forward_impute(input_tensor)

    def test_fill_check(self):
        """Test check for ``fill`` argument."""
        # Argument required
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
        # Argument not required
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

    def test_initial_missing(self, tensor_missing):
        """Test fill of initial missing values."""
        test_tensor = forward_impute(tensor_missing, fill=torch.tensor([111.0, 222.0]))
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

    def test_select_fill(self, tensor_missing):
        """Test ``select`` with ``fill`` argument."""
        # No fill argument
        with pytest.raises(
            AssertionError, match=re.escape("argument 'fill' must be provided")
        ):
            forward_impute(tensor_missing, select=torch.tensor([0]))
        # Fill argument
        test_tensor = forward_impute(
            tensor_missing, fill=torch.tensor([111.0]), select=torch.tensor([0])
        )
        expect_tensor = torch.tensor(
            [
                [111.0, 2.0],
                [111.0, 4.0],
                [5.0, float("nan")],
                [5.0, 8.0],
                [9.0, float("nan")],
                [11.0, 12.0],
            ]
        )
        assert torch.allclose(
            test_tensor, expect_tensor, rtol=RTOL, atol=ATOL, equal_nan=True
        )

    def test_select(self, tensor_missing):
        """Test ``select`` where ``fill`` argument is not required."""
        test_tensor = forward_impute(tensor_missing, select=torch.tensor([1]))
        expect_tensor = torch.tensor(
            [
                [float("nan"), 2.0],
                [float("nan"), 4.0],
                [5.0, 4.0],
                [float("nan"), 8.0],
                [9.0, 8.0],
                [11.0, 12.0],
            ]
        )
        assert torch.allclose(
            test_tensor, expect_tensor, rtol=RTOL, atol=ATOL, equal_nan=True
        )

    def test_tensor(self, tensor_missing):
        """Test with an example tensor."""
        test_tensor = forward_impute(tensor_missing, fill=torch.tensor([111.0, 222.0]))
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
            seed=SEED,
        )
        # Check no NaNs post imputation
        X_impute = forward_impute(dataset.X, fill=torch.Tensor([float("nan"), 1, 2, 3]))
        assert torch.sum(torch.isnan(X_impute)).item() == 0
