"""
========================
Custom collate functions
========================

Custom collate functions should be passed to the ``collate_fn`` argument of a
`DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_:

* `sort_by_length() <torchtime.collate.sort_by_length>`_ sorts each batch by descending
  length

* `packed_sequence() <torchtime.collate.packed_sequence>`_ returns ``X`` and ``y`` as a
  ``PackedSequence`` object
"""

from typing import Dict, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence


def sort_by_length(batch_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """**Collates a batch and sorts by descending length.**

    Pass to the ``collate_fn`` argument when creating a PyTorch DataLoader. Batches are
    a named dictionary with ``X``, ``y`` and ``length`` data that is sorted by
    length. ``pack_padded_sequence()`` can therefore be called in the forward method of
    the model. For example:

    .. testcode::

        from torch.utils.data import DataLoader
        from torchtime.data import UEA
        from torchtime.collate import sort_by_length

        char_traj = UEA(
            dataset="CharacterTrajectories",
            split="train",
            train_prop=0.7,
            seed=123,
        )
        dataloader = DataLoader(
            char_traj,
            batch_size=32,
            collate_fn=sort_by_length,
        )
        print(next(iter(dataloader))["length"])

    .. testoutput::

        ...
        tensor([157, 151, 151, 150, 138, 138, 136, 135, 133, 130, 129, 129, 127, 126,
                124, 124, 121, 121, 118, 117, 117, 117, 113, 108, 106, 105, 102,  98,
                 83,  74,  74,  61])

    Args:
        batch_data: Batch from a ``torchtime.data`` class.

    Returns:
        Updated batch.
    """
    X = torch.stack([i["X"] for i in batch_data], dim=0)
    y = torch.stack([i["y"] for i in batch_data])
    length = torch.stack([i["length"] for i in batch_data])
    length, sorted_index = torch.sort(length, descending=True)
    X = torch.index_select(X, 0, sorted_index)
    y = torch.index_select(y, 0, sorted_index)
    return {"X": X, "y": y, "length": length}


def packed_sequence(
    batch_data: Dict[str, Tensor]
) -> Dict[str, Union[PackedSequence, Tensor]]:
    """**Collates a batch and returns data as PackedSequence objects.**

    Pass to the ``collate_fn`` argument when creating a PyTorch DataLoader. Batches are
    a named dictionary with ``X``, ``y`` and ``length`` data where ``X`` and ``y`` are
    PackedSequence objects. For example:

    .. testcode::

        from torch.utils.data import DataLoader
        from torchtime.data import UEA
        from torchtime.collate import packed_sequence

        char_traj = UEA(
            dataset="CharacterTrajectories",
            split="train",
            train_prop=0.7,
            seed=123,
        )
        dataloader = DataLoader(
            char_traj,
            batch_size=32,
            collate_fn=packed_sequence,
        )
        print(next(iter(dataloader))["X"])

    .. testoutput::

        ...
        PackedSequence(data=tensor([[ 0.0000e+00,  2.2753e-01,  6.0560e-03,  2.0894e-02],
                [ 0.0000e+00, -3.6401e-02,  1.1512e-01,  7.3964e-01],
                [ 0.0000e+00,  5.6454e-01, -1.0000e-05,  2.9244e-01],
                ...,
                [ 1.5400e+02, -2.6396e-01,  1.9185e-01, -1.4082e+00],
                [ 1.5500e+02, -2.2807e-01,  1.6577e-01, -1.2167e+00],
                [ 1.5600e+02, -1.7705e-01,  1.2868e-01, -9.4452e-01]]), batch_sizes=tensor([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
                31, 31, 29, 29, 29, 29, 29, 29, 29, 29, 29, 28, 28, 28, 28, 28, 28, 28,
                28, 28, 28, 28, 28, 28, 28, 28, 27, 27, 27, 27, 26, 26, 26, 25, 24, 24,
                23, 23, 23, 23, 23, 22, 22, 22, 22, 19, 18, 18, 18, 16, 16, 16, 14, 14,
                13, 12, 12, 10,  9,  9,  9,  8,  8,  7,  6,  6,  4,  4,  4,  4,  4,  4,
                 4,  4,  4,  4,  4,  4,  3,  1,  1,  1,  1,  1,  1]), sorted_indices=None, unsorted_indices=None)

    Args:
        batch_data: Batch from a ``torchtime.data`` class.

    Returns:
        Updated batch.
    """  # noqa: E501
    batch_data = sort_by_length(batch_data)
    length = batch_data["length"]
    X = pack_padded_sequence(batch_data["X"], length, batch_first=True)
    y = pack_padded_sequence(batch_data["y"], length, batch_first=True)
    return {"X": X, "y": y, "length": length}
