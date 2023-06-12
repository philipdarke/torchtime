import os

import pytest
import torch

from torchtime.constants import CHECKSUM_EXT, OBJ_EXT
from torchtime.utils import _check_SHA256, _get_SHA256

SEED = 456789
N = 10000
S = 350
C = 10


class TestUtils:
    """Unit tests for utility functions."""

    @pytest.fixture
    def checksum_files(self):
        """Fixture for checksum tests."""
        generator = torch.Generator().manual_seed(SEED)
        X = torch.rand((N, S, C), generator=generator)
        sha = "3460c01e7ab9d6bfab46c1802e52362f997bf03c1723c7cf7fc70cd5df0de780"
        sha_error = "1111111111111111111111111111111111111111111111111111111111111111"
        X_path = ".torchtime/X" + OBJ_EXT
        sha_path = ".torchtime/correct" + CHECKSUM_EXT
        error_path = ".torchtime/error" + CHECKSUM_EXT
        # Save files
        torch.save(X, X_path)
        with open(sha_path, "w") as f:
            f.write(sha)
        with open(error_path, "w") as f:
            f.write(sha_error)
        yield X_path, sha_path, error_path, sha, sha_error
        # Remove files
        os.remove(X_path)
        os.remove(sha_path)
        os.remove(error_path)

    def test_get_sha(self, checksum_files):
        """Test _get_sha() function."""
        X_path, _, _, sha, sha_error = checksum_files
        assert _get_SHA256(X_path) == sha
        assert not _get_SHA256(X_path) == sha_error

    def test_check_sha(self, checksum_files):
        """Test check_sha() function."""
        X_path, sha_path, error_path, _, _ = checksum_files
        assert _check_SHA256(X_path, sha_path)
        assert not _check_SHA256(X_path, error_path)
