import torch
from torch.nn.utils.rnn import pack_padded_sequence


def sort_by_length(batch_data):
    """Collates a batch and sorts by decending length.

    If passed to a DataLoader, the batch is sorted by length and
    ``pack_padded_sequence()`` can be called in the forward method of the model.

    Example::

        from torch.utils.data import DataLoader
        from torchtime.data import UEA
        from torchtime.collate import sort_by_length

        char_traj = UEA(
            dataset="CharacterTrajectories",
            split="train",
            train_split=0.7,
            seed=456789,
        )
        dataloader = DataLoader(
            char_traj,
            batch_size=32,
            collate_fn=sort_by_length,
        )

        >> next(iter(dataloader))["length"]

        tensor([182, 155, 148, 147, 145, 141, 139, 138, 138, 137, 137, 136, 135, 131,
                131, 130, 129, 124, 123, 121, 115, 114, 110, 110, 107, 104,  94,  94,
                92,  86,  73,  73])
    """
    X = torch.stack([i["X"] for i in batch_data], dim=0)
    y = torch.stack([i["y"] for i in batch_data])
    length = torch.stack([i["length"] for i in batch_data])
    length, sorted_index = torch.sort(length, descending=True)
    X = torch.index_select(X, 0, sorted_index)
    y = torch.index_select(y, 0, sorted_index)
    return {"X": X, "y": y, "length": length}


def packed_sequence(batch_data):
    """Collates a batch and returns a PackedSequence.

    If passed to a DataLoader, ``X`` is returned as a PaddedSequence object.

    Example::

        from torch.utils.data import DataLoader
        from torchtime.data import UEA
        from torchtime.collate import packed_sequence

        char_traj = UEA(
            dataset="CharacterTrajectories",
            split="train",
            train_split=0.7,
            seed=456789,
        )
        dataloader = DataLoader(
            char_traj,
            batch_size=32,
            collate_fn=packed_sequence,
        )

        >> next(iter(dataloader))["X"]

        PackedSequence(
            data=tensor([[ 0.0000e+00,  2.1401e-01,  1.6085e-01,  3.3961e-01],
                         [ 0.0000e+00,  6.4870e-03, -3.4780e-02,  2.3045e-01],
                         [ 0.0000e+00,  2.1153e-01, -7.7158e-02, -8.6610e-03],
                         ...,
                         [ 1.7900e+02,  3.5498e-01,  5.1161e-01, -1.7406e+00],
                         [ 1.8000e+02,  3.0671e-01,  4.4205e-01, -1.5039e+00],
                         [ 1.8100e+02,  2.3810e-01,  3.4316e-01, -1.1675e+00]]),
            batch_sizes=tensor([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
                                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                                32, 32, 32, 32, 32, 32, 32, 32, 30, 30, 30, 30, 30,
                                30, 30, 30, 30, 30, 30, 30, 30, 29, 29, 29, 29, 29,
                                29, 28, 28, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
                                25, 25, 25, 24, 24, 24, 22, 22, 22, 22, 21, 20, 20,
                                20, 20, 20, 20, 19, 19, 18, 17, 17, 17, 17, 17, 16,
                                15, 13, 13, 13, 13, 12, 11,  9,  7,  6,  6,  5,  5,
                                 5,  5,  4,  4,  3,  2,  2,  2,  2,  2,  2,  2,  1,
                                 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                                 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]),
            sorted_indices=None,
            unsorted_indices=None
        )
    """
    batch_data = sort_by_length(batch_data)
    length = batch_data["length"]
    X = pack_padded_sequence(batch_data["X"], length, batch_first=True)
    return {"X": X, "y": batch_data["y"], "length": length}
