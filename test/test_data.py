import pytest
import torch

from torchtime.data import UEA

# class TestPhysioNet2019:
#     """GITHUB ACTIONS RUNNER HAS INSUFFICIENT MEMORY FOR THIS TEST"""
#
#     def test_dataset(self):
#         dataset = PhysioNet2019(
#             split="test", train_split=0.7, val_split=0.2, seed=456789
#         )
#         # Check data set size
#         assert dataset.X_train.shape == torch.Size([28236, 336, 41])
#         assert dataset.X_val.shape == torch.Size([8067, 336, 41])
#         assert dataset.X_test.shape == torch.Size([4033, 336, 41])
#         # Check correct split is returned
#         assert torch.allclose(dataset.X, dataset.X_test, equal_nan=True)
#         assert torch.allclose(dataset.y, dataset.y_test, equal_nan=True)
#         assert torch.allclose(dataset.length, dataset.length_test, equal_nan=True)
#
#     def test_dataset_mask_delta(self):
#         dataset = PhysioNet2019(
#             split="test",
#             train_split=0.7,
#             val_split=0.2,
#             time=False,
#             mask=True,
#             delta=True,
#             seed=456789,
#         )
#         # Check data set size
#         assert dataset.X_train.shape == torch.Size([28236, 336, 120])
#         assert dataset.X_val.shape == torch.Size([8067, 336, 120])
#         assert dataset.X_test.shape == torch.Size([4033, 336, 120])
#         # Check correct split is returned
#         assert torch.allclose(dataset.X, dataset.X_test, equal_nan=True)
#         assert torch.allclose(dataset.y, dataset.y_test, equal_nan=True)
#         assert torch.allclose(dataset.length, dataset.length_test, equal_nan=True)


class TestUEA:
    def test_dataset_no_time_mask(self):
        dataset = UEA(
            dataset="ArrowHead",
            split="test",
            missing=0.5,
            train_split=0.7,
            val_split=0.2,
            time=False,
            seed=456789,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([148, 251, 1])
        assert dataset.X_val.shape == torch.Size([42, 251, 1])
        assert dataset.X_test.shape == torch.Size([21, 251, 1])
        # Check correct split is returned
        assert torch.allclose(dataset.X, dataset.X_test, equal_nan=True)
        assert torch.allclose(dataset.y, dataset.y_test, equal_nan=True)
        assert torch.allclose(dataset.length, dataset.length_test, equal_nan=True)
        # Check missing data
        assert (
            torch.sum(torch.isnan(dataset.X_train)).item() == 18500
        )  # expect around 148 * 251 * 0.5 = 18,574
        assert (
            torch.sum(torch.isnan(dataset.X_val)).item() == 5250
        )  # expect around 42 * 251 * 0.5 = 5,271
        assert (
            torch.sum(torch.isnan(dataset.X_test)).item() == 2625
        )  # expect around 21 * 251 * 0.5 = 2,535

    def test_dataset_alt_splits(self):
        dataset = UEA(
            dataset="ArrowHead",
            split="train",
            train_split=0.7,
            seed=456789,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([148, 251, 2])
        assert dataset.X_val.shape == torch.Size([63, 251, 2])
        # Ensure no test data is returned
        with pytest.raises(AttributeError):
            dataset.X_test
        with pytest.raises(AttributeError):
            dataset.y_test
        with pytest.raises(AttributeError):
            dataset.length_test
        # Check correct split is returned
        assert torch.allclose(dataset.X, dataset.X_train, equal_nan=True)
        assert torch.allclose(dataset.y, dataset.y_train, equal_nan=True)
        assert torch.allclose(dataset.length, dataset.length_train, equal_nan=True)

    def test_dataset_missing_by_channel_mask(self):
        dataset = UEA(
            dataset="CharacterTrajectories",
            split="val",
            missing=[0.1, 0.5, 0.9],
            train_split=0.7,
            val_split=0.2,
            mask=True,
            seed=456789,
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
            torch.sum(torch.isnan(dataset.X_train)).item() == 731692
        )  # expect around (2002 * 182 - 239917) * 3 + 239917 * 1.5 = 733,217
        assert (
            torch.sum(torch.isnan(dataset.X_val)).item() == 207946
        )  # expect around (571 * 182 - 68934) * 3 + 68934 * 1.5 = 208,365
        assert (
            torch.sum(torch.isnan(dataset.X_test)).item() == 104184
        )  # expect around (285 * 182 - 34088) * 3 + 34088 * 1.5 = 104,478
