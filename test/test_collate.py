import torch
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import DataLoader

from torchtime.collate import packed_sequence, sort_by_length
from torchtime.data import UEA


class TestCollateFunctions:
    def test_sort_by_length(self):
        dataset = UEA(
            dataset="CharacterTrajectories",
            split="train",
            train_prop=0.7,
            val_prop=0.2,
        )
        loader = DataLoader(
            dataset, batch_size=32, shuffle=True, collate_fn=sort_by_length
        )
        next_batch = next(iter(loader))
        # Check batch lengths are sorted
        sorted_length, _ = torch.sort(next_batch["length"], descending=True)
        assert torch.equal(sorted_length, next_batch["length"])

    def test_packed_sequence(self):
        dataset = UEA(
            dataset="CharacterTrajectories",
            split="train",
            train_prop=0.7,
            val_prop=0.2,
        )
        loader = DataLoader(
            dataset, batch_size=32, shuffle=True, collate_fn=packed_sequence
        )
        next_batch = next(iter(loader))
        # Check X is a PackedSequence object
        assert type(next_batch["X"]) is PackedSequence
