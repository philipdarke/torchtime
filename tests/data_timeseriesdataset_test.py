import pytest

from torchtime.data import _TimeSeriesDataset


class TestTimeSeriesDataset:
    """Test _TimeSeriesDataset class."""

    def test_base_class(self):
        """Fail on using private class."""
        with pytest.raises(NotImplementedError):
            _TimeSeriesDataset(
                dataset="test",
                split="train",
                train_prop=0.7,
            )
