# Custom collate functions

Custom collate functions should be passed to the `collate_fn` argument of a [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

Data sets of variable length can be efficiently represented in PyTorch using a [`PackedSequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html) object. These are formed using
[`pack_padded_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence) which by default expects the input batch to be sorted in descending length. Two collate functions are provided to support the use of `PackedSequence` objects in models:

* [`sort_by_length()`](torchtime.collate.sort_by_length) sorts each batch by descending length.

* [`packed_sequence()`](torchtime.collate.packed_sequence) returns `X` and `y` as a `PackedSequence` object.

```{eval-rst}
.. automodule:: torchtime.collate
   :members: 
```