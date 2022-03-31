# Tutorial

This tutorial covers the use of the `torchtime.data.UEA` class for data sets held in the UEA/UCR classification repository [[link]](https://www.timeseriesclassification.com/). A similar approach applies for other data sets, see the [API](api) for further details.

## Basic usage

`torchtime.data.UEA` has the following basic arguments:

* The data set is specified using the `dataset` argument (see list [here](https://www.timeseriesclassification.com/dataset.php)).
* The `split` argument determines whether training, validation or test data are returned.
* The size of the splits are controlled with the `train_split` and `val_split` arguments. For a training/validation split use `train_split` only. For a training/validation/test split specify both `train_split` and `val_split`.
* For reproducibility, an optional random `seed` can be specified.

For example, to load training data for the [ArrowHead](https://www.timeseriesclassification.com/description.php?Dataset=ArrowHead) data set with a 70/30% training/validation split:

```
from torch.utils.data import DataLoader
from torchtime.data import UEA

arrowhead = UEA(
    dataset="ArrowHead",
    split="train",
    train_split=0.7,
    seed=456789,
)
dataloader = DataLoader(arrowhead, batch_size=32)
```

The DataLoader returns batches as a dictionary of tensors `X`, `y` and `length`. `X` are the time series data. By default, a time stamp is appended to the data as the first channel. This package follows the *batch first* convention therefore `X` has shape (*n*, *s*, *c*) where *n* is batch size, *s* is trajectory length and *c* is the number of channels.

ArrowHead is a univariate time series therefore `X` has two channels, the time stamp followed by the time series.

```
>> next(iter(dataloader))["X"][0, 0:10]  # first 10 observations of the first trajectory

tensor([[ 0.0000, -1.8302],
        [ 1.0000, -1.8123],
        [ 2.0000, -1.8122],
        [ 3.0000, -1.7655],
        [ 4.0000, -1.7484],
        [ 5.0000, -1.7128],
        [ 6.0000, -1.6731],
        [ 7.0000, -1.6115],
        [ 8.0000, -1.5760],
        [ 9.0000, -1.5368]])
```

The time stamp can be removed with the `time` argument:

```
arrowhead = UEA(
    dataset="ArrowHead",
    split="train",
    train_split=0.7,
    time=False,
    seed=456789,
)
dataloader = DataLoader(arrowhead, batch_size=32)

>> next(iter(dataloader))["X"][0, 0:10]

tensor([[-1.8302],
        [-1.8123],
        [-1.8122],
        [-1.7655],
        [-1.7484],
        [-1.7128],
        [-1.6731],
        [-1.6115],
        [-1.5760],
        [-1.5368]])
```

## Simulating missing data

Most UEA/UCR data sets are regularly sampled and fully observed. Missing data can be simulated using the `missing` argument i.e. the probability that data are missing. Data are dropped at random.

### Regularly sampled with missing time points

If `missing` is a single value, data are dropped across all channels. This simulates regularly sampled data where some time points are not recorded. Using the [CharacterTrajectories](http://timeseriesclassification.com/description.php?Dataset=CharacterTrajectories) data set as an example:

```
char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_split=0.7,
    missing=0.5,
    seed=456789,
)
dataloader = DataLoader(char_traj, batch_size=32)

>> next(iter(dataloader))["X"][0, 0:10]

tensor([[ 0.0000,     nan,     nan,     nan],
        [ 1.0000,  0.0535,  0.1563,  1.2596],
        [ 2.0000,     nan,     nan,     nan],
        [ 3.0000, -0.0670,  0.0809,  1.6217],
        [ 4.0000,     nan,     nan,     nan],
        [ 5.0000,     nan,     nan,     nan],
        [ 6.0000, -0.3238, -0.2532,  1.5915],
        [ 7.0000,     nan,     nan,     nan],
        [ 8.0000,     nan,     nan,     nan],
        [ 9.0000, -0.5236, -0.7445,  1.2118]])
```

### Regularly sampled and partially observed

Alternatively, data can be dropped independently for each channel by passing a list representing the proportion missing for each channel. This simulates regularly sampled data with partial observation i.e. not all channels are recorded at each time point.

```
char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_split=0.7,
    missing=[0.8, 0.2, 0.5],
    seed=456789,
)
dataloader = DataLoader(char_traj, batch_size=32)

>> next(iter(dataloader))["X"][0, 0:10]

tensor([[ 0.0000,     nan,  0.1423,     nan],
        [ 1.0000,  0.0535,  0.1563,     nan],
        [ 2.0000,     nan,  0.1365,  1.4829],
        [ 3.0000,     nan,  0.0809,  1.6217],
        [ 4.0000,     nan, -0.0062,     nan],
        [ 5.0000,     nan, -0.1191,     nan],
        [ 6.0000,     nan, -0.2532,  1.5915],
        [ 7.0000, -0.3965, -0.4040,     nan],
        [ 8.0000,     nan,     nan,  1.3501],
        [ 9.0000,     nan, -0.7445,     nan]])
```

Note that each time point has a varying number of observations.

## Missing data masks

In some applications, the presence (or absence) of data can itself be informative. For example, a doctor may be more likely to order a particular diagnostic test if they believe the patient has a medical condition. Missing data/observational masks can be used to inform models of missing data. These are appended by setting `mask` to `True`.

```
char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_split=0.7,
    missing=[0.8, 0.2, 0.5],
    mask=True,
    seed=456789
)
dataloader = DataLoader(char_traj, batch_size=32)

>> next(iter(dataloader))["X"][0, 0:10]

tensor([[ 0.0000,     nan,  0.1423,     nan,  0.0000,  1.0000,  0.0000],
        [ 1.0000,  0.0535,  0.1563,     nan,  1.0000,  1.0000,  0.0000],
        [ 2.0000,     nan,  0.1365,  1.4829,  0.0000,  1.0000,  1.0000],
        [ 3.0000,     nan,  0.0809,  1.6217,  0.0000,  1.0000,  1.0000],
        [ 4.0000,     nan, -0.0062,     nan,  0.0000,  1.0000,  0.0000],
        [ 5.0000,     nan, -0.1191,     nan,  0.0000,  1.0000,  0.0000],
        [ 6.0000,     nan, -0.2532,  1.5915,  0.0000,  1.0000,  1.0000],
        [ 7.0000, -0.3965, -0.4040,     nan,  1.0000,  1.0000,  0.0000],
        [ 8.0000,     nan,     nan,  1.3501,  0.0000,  0.0000,  1.0000],
        [ 9.0000,     nan, -0.7445,     nan,  0.0000,  1.0000,  0.0000]])
```

Note the final three channels indicate whether data were recorded.

## Time deltas

Some models use the time since the previous observation as an input e.g. GRU-D. This can be added using the `delta` argument. See [Che et al, 2018](https://doi.org/10.1038/s41598-018-24271-9) for implementation details.

```
char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_split=0.7,
    missing=[0.8, 0.2, 0.5],
    delta=True,
    seed=456789
)
dataloader = DataLoader(char_traj, batch_size=32)

>> next(iter(dataloader))["X"][0, 0:10]

tensor([[ 0.0000,     nan,  0.1423,     nan,  0.0000,  0.0000,  0.0000],
        [ 1.0000,  0.0535,  0.1563,     nan,  1.0000,  1.0000,  1.0000],
        [ 2.0000,     nan,  0.1365,  1.4829,  1.0000,  1.0000,  2.0000],
        [ 3.0000,     nan,  0.0809,  1.6217,  2.0000,  1.0000,  1.0000],
        [ 4.0000,     nan, -0.0062,     nan,  3.0000,  1.0000,  1.0000],
        [ 5.0000,     nan, -0.1191,     nan,  4.0000,  1.0000,  2.0000],
        [ 6.0000,     nan, -0.2532,  1.5915,  5.0000,  1.0000,  3.0000],
        [ 7.0000, -0.3965, -0.4040,     nan,  6.0000,  1.0000,  1.0000],
        [ 8.0000,     nan,     nan,  1.3501,  1.0000,  1.0000,  2.0000],
        [ 9.0000,     nan, -0.7445,     nan,  2.0000,  2.0000,  1.0000]])
```

The first channel is observed at times 1 and 7 therefore the time delta is 6 at time 7. Note that the time delta is 0 at time 0 by definition.

## Combining output options

The `time`, `mask` and `delta` arguments can be combined as required:

```
char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_split=0.7,
    missing=[0.8, 0.2, 0.5],
    # time=False,
    mask=True,
    delta=True,
    seed=456789
)
dataloader = DataLoader(char_traj, batch_size=32)

>> next(iter(dataloader))["X"][0, 0:10]

tensor([[    nan,  0.1423,     nan,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0535,  0.1563,     nan,  1.0000,  1.0000,  0.0000,  1.0000,  1.0000,  1.0000],
        [    nan,  0.1365,  1.4829,  0.0000,  1.0000,  1.0000,  1.0000,  1.0000,  2.0000],
        [    nan,  0.0809,  1.6217,  0.0000,  1.0000,  1.0000,  2.0000,  1.0000,  1.0000],
        [    nan, -0.0062,     nan,  0.0000,  1.0000,  0.0000,  3.0000,  1.0000,  1.0000],
        [    nan, -0.1191,     nan,  0.0000,  1.0000,  0.0000,  4.0000,  1.0000,  2.0000],
        [    nan, -0.2532,  1.5915,  0.0000,  1.0000,  1.0000,  5.0000,  1.0000,  3.0000],
        [-0.3965, -0.4040,     nan,  1.0000,  1.0000,  0.0000,  6.0000,  1.0000,  1.0000],
        [    nan,     nan,  1.3501,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000,  2.0000],
        [    nan, -0.7445,     nan,  0.0000,  1.0000,  0.0000,  2.0000,  2.0000,  1.0000]])
```

Here, the initial time channel is not returned, but the missing data and time delta channels are appened to the data.

## Label data

Labels `y` are one-hot encoded and have shape (*n*, *l*) where *l* is the number of classes. For example, CharacterTrajectories has 20 classes.

```
>> next(iter(dataloader))["y"]

tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        ...,

        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
```

## Trajectory lengths

The `length` of each trajectory (before padding if the data set is of irregular length) is provided as a tensor of shape (*n*). For example, for CharacterTrajectories:

```
>> next(iter(dataloader))["length"]

tensor([124, 131, 110,  92, 115, 131, 130,  86, 155, 141, 129, 114, 135,  73,
        138, 147, 182, 107,  94, 110,  94, 136,  73, 123, 137, 148, 138, 104,
        121, 137, 145, 139])
```

Trajectories are padded with `NaNs` to the length of the longest trajectory if the data set is of irregular length.

```
char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_split=0.7,
    missing=[0.8, 0.2, 0.5],
    mask=True,
    delta=True,
    seed=456789
)
dataloader = DataLoader(char_traj, batch_size=32)

>> next(iter(dataloader))["X"][0]  # first trajectory

tensor([[   0.0000,     nan,  0.1423,     nan,  0.0000,  1.0000,  0.0000,   0.0000,   0.0000,   0.0000],
        [   1.0000,  0.0535,  0.1563,     nan,  1.0000,  1.0000,  0.0000,   1.0000,   1.0000,   1.0000],
        [   2.0000,     nan,  0.1365,  1.4829,  0.0000,  1.0000,  1.0000,   1.0000,   1.0000,   2.0000],

        ...,

        [ 179.0000,     nan,     nan,     nan,  0.0000,  0.0000,  0.0000,  62.0000,  56.0000,  56.0000],
        [ 180.0000,     nan,     nan,     nan,  0.0000,  0.0000,  0.0000,  63.0000,  57.0000,  57.0000],
        [ 181.0000,     nan,     nan,     nan,  0.0000,  0.0000,  0.0000,  64.0000,  58.0000,  58.0000]])
```

Note the end of trajectory has been padded with `NaNs`.

## PackedSequence objects

Data sets of variable length can be efficiently represented in PyTorch using a [`PackedSequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html) object. These are formed using
[`pack_padded_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence) which by default expects the input batch to be sorted in descending length. This is handled by passing the [`torchtime.collate.sort_by_length`](torchtime.collate.sort_by_length) function to the `collate_fn` DataLoader argument.

```
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
```

The batch is now sorted by length and ``pack_padded_sequence()`` can be called in the forward method of the model.

Alternatively, if your model handles a `PackedSequence` directly (all Pytorch RNNs do so), use the [`torchtime.collate.packed_sequence`](torchtime.collate.packed_sequence) function.

```
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
```