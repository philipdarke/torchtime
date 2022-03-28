import torch


def sort_by_length(batch_data):
    """Collates a batch and sorts by decending length.

    Data sets of variable length can be efficiently represented in a
    `PackedSequence
    <https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html>`_
    object. These are formed using
    `pack_padded_sequence()
    <https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence>`_
    which by default expects the input batch to be sorted by decending length. Pass this
    collate function to a DataLoader to automatically sort the batch before calling
    pack_padded_sequence() in the forward method of your model.

    Example:
    The CharacterTrajectories data set features trajectories of varying length. The
    ``sort_by_length()`` collate function sorts the batch::

        from torch.utils.data import DataLoader
        from torchtime.data import UEA
        from torchtime.collate import sort_by_length

        char_traj = UEA(
            dataset="CharacterTrajectories",
            split="train",
            train_split=0.7,
            val_split=0.2,
        )
        dataloader = DataLoader(
            char_traj,
            batch_size=32,
            collate_fn=sort_by_length,
        )

        >> next(iter(dataloader))["length"]

        tensor([146, 145, 143, 136, 136, 135, 132, 131, 131, 130, 130, 129, 125, 125,
                124, 124, 122, 121, 119, 119, 116, 115, 115, 114, 111, 109, 106, 105,
                103,  98,  88,  78])
    """
    X = torch.stack([i["X"] for i in batch_data], dim=0)
    y = torch.stack([i["y"] for i in batch_data])
    length = torch.stack([i["length"] for i in batch_data])
    length, sorted_index = torch.sort(length, descending=True)
    X = torch.index_select(X, 0, sorted_index)
    y = torch.index_select(y, 0, sorted_index)
    return {"X": X, "y": y, "length": length}
