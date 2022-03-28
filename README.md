# Time series data sets for PyTorch

![PyPi](https://img.shields.io/pypi/v/torchtime)
[![Build status](https://img.shields.io/github/workflow/status/philipdarke/torchtime/build.svg)](https://github.com/philipdarke/torchtime/actions/workflows/build.yml)
![Coverage](https://philipdarke.com/torchtime/assets/coverage-badge.svg)
[![License](https://img.shields.io/github/license/philipdarke/torchtime.svg)](https://github.com/philipdarke/torchtime/blob/main/LICENSE)

`torchtime` provides ready-to-go time series data sets for use in PyTorch. The current list of supported data sets is:

* All data sets in the UEA/UCR classification repository [[link]](https://www.timeseriesclassification.com/)
* PhysioNet Challenge 2019 [[link]](https://physionet.org/content/challenge-2019/1.0.0/)

The package follows the *batch first* convention. Data tensors are therefore of shape (*n*, *s*, *c*) where *n* is batch size, *s* is trajectory length and *c* are the number of channels.

## Installation

```bash
$ pip install torchtime
```

## Using `torchtime`

The example below uses the `torchtime.data.UEA` class. The data set is specified using the `dataset` argument (see list [here](https://www.timeseriesclassification.com/dataset.php)). The `split` argument determines whether training, validation or test data are returned. The size of the splits are controlled with the `train_split` and `val_split` arguments.

For example, to load training data for the [ArrowHead](https://www.timeseriesclassification.com/description.php?Dataset=ArrowHead) data set with a 70% training, 20% validation and 10% testing split:

```
from torch.utils.data import DataLoader
from torchtime.data import UEA

arrowhead = UEA(
    dataset="ArrowHead",
    split="train",
    train_split=0.7,
    val_split=0.2,
)
dataloader = DataLoader(arrowhead, batch_size=32)
```

Batches are dictionaries of tensors `X`, `y` and `length`. `X` are the time series data with an additional time stamp in the first channel, `y` are one-hot encoded labels and `length` are the length of each trajectory.

ArrowHead is a univariate time series with 251 observations in each trajectory. `X` therefore has two channels, the time stamp followed by the time series. A batch size of 32 was specified above therefore `X` has shape (32, 251, 2).

```
>> next(iter(dataloader))["X"].shape

torch.Size([32, 251, 2])

>> next(iter(dataloader))["X"]

tensor([[[  0.0000,  -1.8295],
         [  1.0000,  -1.8238],
         [  2.0000,  -1.8101],
         ...,
         [248.0000,  -1.7759],
         [249.0000,  -1.8088],
         [250.0000,  -1.8110]],

        ...,

        [[  0.0000,  -2.0147],
         [  1.0000,  -2.0311],
         [  2.0000,  -1.9471],
         ...,
         [248.0000,  -1.9901],
         [249.0000,  -1.9913],
         [250.0000,  -2.0109]]])
```

There are three classes therefore `y` has shape (32, 3).

```
>> next(iter(dataloader))["y"].shape

torch.Size([32, 3])

>> next(iter(dataloader))["y"]

tensor([[0, 0, 1],
        ...,
        [1, 0, 0]])
```

Finally, `length` is the length of each trajectory (before any padding for data sets of irregular length) and therefore has shape (32).

```
>> next(iter(dataloader))["length"].shape

torch.Size([32])

>> next(iter(dataloader))["length"]

tensor([251, ..., 251])
```

## Learn more

Other features include missing data simulation for UEA data sets. See the [API](api) for more information.

This work is based on some of the data processing ideas in Kidger et al, 2020 [[link]](https://arxiv.org/abs/2005.08926) and Che et al, 2018 [[link]](https://doi.org/10.1038/s41598-018-24271-9).

## License

Released under the MIT license.
