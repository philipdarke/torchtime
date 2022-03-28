import torch

from torchtime.data import UEA

# class TestPhysioNet2019:
#     '''GITHUB ACTIONS RUNNER HAS INSUFFICIENT MEMORY FOR THIS TEST'''
#     def test_dataset(self):
#         dataset = PhysioNet2019(
#             split="test", train_split=0.7, val_split=0.2, seed=456789
#         )
#         # Check data set size
#         assert dataset.X_train.shape == torch.Size([28236, 336, 40])
#         assert dataset.X_val.shape == torch.Size([8067, 336, 40])
#         assert dataset.X_test.shape == torch.Size([4033, 336, 40])
#         # Check correct split is returned
#         assert torch.allclose(dataset.X, dataset.X_test, equal_nan=True)
#         assert torch.allclose(dataset.y, dataset.y_test, equal_nan=True)
#         assert torch.allclose(dataset.length, dataset.length_test, equal_nan=True)


class TestUEA:
    def test_dataset(self):
        dataset = UEA(
            dataset="ArrowHead",
            split="train",
            missing=0.5,
            train_split=0.7,
            val_split=0.2,
            seed=456789,
        )
        # Check data set size
        assert dataset.X_train.shape == torch.Size([148, 251, 2])
        assert dataset.X_val.shape == torch.Size([42, 251, 2])
        assert dataset.X_test.shape == torch.Size([21, 251, 2])
        # Check correct split is returned
        assert torch.allclose(dataset.X, dataset.X_train, equal_nan=True)
        assert torch.allclose(dataset.y, dataset.y_train, equal_nan=True)
        assert torch.allclose(dataset.length, dataset.length_train, equal_nan=True)
