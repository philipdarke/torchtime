# Time series of unequal lengths

The `length` of each trajectory is provided as a tensor of shape (*n*). Some data sets, for example [CharacterTrajectories](http://timeseriesclassification.com/description.php?Dataset=CharacterTrajectories), feature sequences of unequal length:

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_prop=0.7,
    seed=123,
)
dataloader = DataLoader(char_traj, batch_size=32)
next(iter(dataloader))["length"]
```

Output:

```python
tensor([150, 136, 124, 108,  61, 157, 113, 133,  74, 121, 129, 138, 102, 130,
         83, 124, 117, 117, 117, 151, 129, 127, 126, 135,  98, 105, 121, 151,
        106, 118, 138,  74])
```

Trajectories are padded with `NaNs` to the length of the longest trajectory if the data set is of irregular length:

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_prop=0.7,
    seed=123,
)
dataloader = DataLoader(char_traj, batch_size=32)
next(iter(dataloader))["X"][0]
```

Output:

```python
tensor([[ 0.0000e+00, -1.8494e-01,  1.9776e-01,  3.2627e-01],
        [ 1.0000e+00, -2.7413e-01,  2.3993e-01,  3.9900e-01],
        [ 2.0000e+00, -3.7444e-01,  2.5107e-01,  4.2597e-01],
        ...,
        [ 1.4700e+02,  1.4563e-01,  3.0977e-01, -1.4867e+00],
        [ 1.4800e+02,  1.2583e-01,  2.6765e-01, -1.2845e+00],
        [ 1.4900e+02,  9.7681e-02,  2.0777e-01, -9.9716e-01],
        [ 1.5000e+02,         nan,         nan,         nan],
        [ 1.5100e+02,         nan,         nan,         nan],
        [ 1.5200e+02,         nan,         nan,         nan],
        ...,
        [ 1.7900e+02,         nan,         nan,         nan],
        [ 1.8000e+02,         nan,         nan,         nan],
        [ 1.8100e+02,         nan,         nan,         nan]])
```

Note the time series has been padded with `NaNs` from *t* = 150.

## PackedSequence objects

Data sets of variable length can be efficiently represented in PyTorch using a [`PackedSequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html) object. These are formed using
[`pack_padded_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence) which by default expects the input batch to be sorted in descending length. Two collate functions are provided to support the use of `PackedSequence` objects in models:

* [`sort_by_length()`](torchtime.collate.sort_by_length) sorts each batch by descending length.

* [`packed_sequence()`](torchtime.collate.packed_sequence) returns `X` and `y` as a `PackedSequence` object.

Custom collate functions should be passed to the `collate_fn` argument of a [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

```{eval-rst}
.. note::
   All Pytorch RNN modules accept packed sequences as inputs.
```

### `sort_by_length()`

```python
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
next(iter(dataloader))["length"]
```

```python
tensor([157, 151, 151, 150, 138, 138, 136, 135, 133, 130, 129, 129, 127, 126,
        124, 124, 121, 121, 118, 117, 117, 117, 113, 108, 106, 105, 102,  98,
         83,  74,  74,  61])
```

Each batch is now sorted by length and ``pack_padded_sequence()`` can be called in the forward method of a model.

### `packed_sequence()`

The [`packed_sequence()`](torchtime.collate.packed_sequence) function returns `X` and `y` as PackedSequence objects within batches.

```
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
next(iter(dataloader))["X"]
```

Output:

```python
PackedSequence(
    data=tensor([[ 0.0000e+00,  2.2753e-01,  6.0560e-03,  2.0894e-02],
                 [ 0.0000e+00, -3.6401e-02,  1.1512e-01,  7.3964e-01],
                 [ 0.0000e+00,  5.6454e-01, -1.0000e-05,  2.9244e-01],
                 ...,
                 [ 1.5400e+02, -2.6396e-01,  1.9185e-01, -1.4082e+00],
                 [ 1.5500e+02, -2.2807e-01,  1.6577e-01, -1.2167e+00],
                 [ 1.5600e+02, -1.7705e-01,  1.2868e-01, -9.4452e-01]]),
    batch_sizes=tensor([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                        ...,
                        18, 16, 16, 16, 14, 14, 13, 12, 12, 10,  9,  9,
                         8,  8,  7,  6,  6,  4,  4,  4,  4,  4,  4,  4,
                         4,  4,  4,  4,  4,  3,  1,  1,  1,  1,  1,  1]),
    sorted_indices=None,
    unsorted_indices=None
)
```
