# Time series data sets for PyTorch

[![PyPi](https://img.shields.io/pypi/v/torchtime)](https://pypi.org/project/torchtime)
[![Build status](https://img.shields.io/github/workflow/status/philipdarke/torchtime/build.svg)](https://github.com/philipdarke/torchtime/actions/workflows/build.yml)
![Coverage](https://philipdarke.com/torchtime/assets/coverage-badge.svg)
[![License](https://img.shields.io/github/license/philipdarke/torchtime.svg)](https://github.com/philipdarke/torchtime/blob/main/LICENSE)

`torchtime` provides ready-to-go time series data sets for use in PyTorch. The current list of supported data sets is:

* All data sets in the UEA/UCR classification repository [[link]](https://www.timeseriesclassification.com/)
* PhysioNet Challenge 2019 (early prediction of sepsis) [[link]](https://physionet.org/content/challenge-2019/1.0.0/)

## Installation

```bash
$ pip install torchtime
```

## Using `torchtime`

The example below uses the `torchtime.data.UEA` class. The data set is specified using the `dataset` argument (see list of data sets [here](https://www.timeseriesclassification.com/dataset.php)). The `split` argument determines whether training, validation or test data are returned. The size of the splits are controlled with the `train_split` and `val_split` arguments. Reproducibility is achieved using the `seed` argument.

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

ArrowHead is a univariate time series with 251 observations in each trajectory. `X` therefore has two channels, the time stamp followed by the time series.

```
>> next(iter(dataloader))["X"]

tensor([[[  0.0000,  -1.8302],
         [  1.0000,  -1.8123],
         [  2.0000,  -1.8122],
         ...,
         [248.0000,  -1.7821],
         [249.0000,  -1.7971],
         [250.0000,  -1.8280]],

        ...,

        [[  0.0000,  -1.8392],
         [  1.0000,  -1.8314],
         [  2.0000,  -1.8125],
         ...,
         [248.0000,  -1.8359],
         [249.0000,  -1.8202],
         [250.0000,  -1.8387]]])
```

Labels `y` are one-hot encoded and have shape (*n*, *l*) where *l* is the number of classes.

```
>> next(iter(dataloader))["y"]

tensor([[0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],

        ...,

        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]])

```

The `length` of each trajectory (before padding if the data set is of irregular length) is provided as a tensor of shape (*n*).

```
>> next(iter(dataloader))["length"]

tensor([251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251,
        251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251,
        251, 251, 251, 251])
```

## Learn more

Missing data can be simulated using the `missing` argument. In addition, missing data/observational masks and time delta channels can be appended using the `mask` and `delta` arguments. See the [tutorial](https://philipdarke.com/torchtime/tutorial.html) and [API](https://philipdarke.com/torchtime/api.html) for more information.

This work is based on some of the data processing ideas in Kidger et al, 2020 [[1]](https://arxiv.org/abs/2005.08926) and Che et al, 2018 [[2]](https://doi.org/10.1038/s41598-018-24271-9).

## References

1. Kidger, P, Morrill, J, Foster, J, *et al*. Neural Controlled Differential Equations for Irregular Time Series. *arXiv* 2005.08926 (2020). [[arXiv]](https://arxiv.org/abs/2005.08926)

1. Che, Z, Purushotham, S, Cho, K, *et al*. Recurrent Neural Networks for Multivariate Time Series with Missing Values. *Sci Rep* 8, 6085 (2018). [[doi]](https://doi.org/10.1038/s41598-018-24271-9)

1. Reyna M, Josef C, Jeter R, *et al*. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge. *Critical Care Medicine* 48 2: 210-217 (2019). [[doi]](https://doi.org/10.1097/CCM.0000000000004145)

1. Reyna, M, Josef, C, Jeter, R, *et al*. Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019 (version 1.0.0). *PhysioNet* (2019). [[doi]](https://doi.org/10.13026/v64v-d857)

1. Goldberger, A, Amaral, L, Glass, L, *et al*. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation* 101 (23), pp. e215–e220 (2000). [[doi]](https://doi.org/10.1161/01.cir.101.23.e215)

## Funding

This work was supported by the Engineering and Physical Sciences Research Council, Centre for Doctoral Training in Cloud Computing for Big Data, Newcastle University (grant number EP/L015358/1).

## License

Released under the MIT license.
