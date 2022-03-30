# API

## Time series data sets

* [PhysioNet2019](torchtime.data.PhysioNet2019)
* [UEA](torchtime.data.UEA)

```{eval-rst}
.. automodule:: torchtime.data
   :members: 
```

## Custom collate functions

Data sets of variable length can be efficiently represented in PyTorch using a [`PackedSequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html) object. These are formed using
[`pack_padded_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence) which by default expects the input batch to be sorted in descending length. This is handled by the [`sort_by_length()`](torchtime.collate.sort_by_length) collate function. Alternatively, a `PackedSequence` object can be formed using the [`packed_sequence()`](torchtime.collate.packed_sequence) collate function.

Custom collate functions should be passed to the `collate_fn` argument of a [DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader).

```{eval-rst}
.. automodule:: torchtime.collate
   :members: 
```
