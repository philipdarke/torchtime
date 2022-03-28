import torch
from torch.utils.data import DataLoader

from torchtime.collate import sort_by_length
from torchtime.data import UEA


class TestCollateFunctions:
    def test_sort_by_length(self):
        """Check batch lengths are sorted."""
        dataset = UEA(
            dataset="CharacterTrajectories",
            split="train",
            missing=0.5,
            train_split=0.7,
            val_split=0.2,
        )
        loader = DataLoader(
            dataset, batch_size=32, shuffle=True, collate_fn=sort_by_length
        )
        next_batch = next(iter(loader))
        sorted_length, _ = torch.sort(next_batch["length"], descending=True)
        assert torch.equal(sorted_length, next_batch["length"])
