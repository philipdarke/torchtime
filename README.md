# Benchmark time series data sets for PyTorch

[![PyPi](https://img.shields.io/pypi/v/torchtime)](https://pypi.org/project/torchtime)
[![Build status](https://img.shields.io/github/workflow/status/philipdarke/torchtime/build.svg)](https://github.com/philipdarke/torchtime/actions/workflows/build.yml)
![Coverage](https://philipdarke.com/torchtime/assets/coverage-badge.svg?dummy=8484744)
[![License](https://img.shields.io/github/license/philipdarke/torchtime.svg)](https://github.com/philipdarke/torchtime/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/475093888.svg)](https://zenodo.org/badge/latestdoi/475093888)

PyTorch data sets for supervised time series classification and prediction problems, including:

* All UEA/UCR classification repository data sets
* PhysioNet Challenge 2012 (in-hospital mortality)
* PhysioNet Challenge 2019 (sepsis prediction)
* A binary prediction variant of the 2019 PhysioNet Challenge

## Why use `torchtime`?

1. Saves time. You don't have to write your own PyTorch data classes.
2. Better research. Use common, reproducible implementations of data sets for a level playing field when evaluating models.

## Installation

```bash
$ pip install torchtime
```

## Getting started

Data classes have a common API. The `split` argument determines whether training ("*train*"), validation ("*val*") or test ("*test*") data are returned. The size of the splits are controlled with the `train_prop` and (optional) `val_prop` arguments.

### PhysioNet data sets

Three [PhysioNet](https://physionet.org/) data sets are currently supported:

* [`torchtime.data.PhysioNet2012`](https://philipdarke.com/torchtime/api/data.html#torchtime.data.PhysioNet2012) returns the 2012 challenge (in-hospital mortality) [[link]](https://physionet.org/content/challenge-2012/1.0.0/).
* [`torchtime.data.PhysioNet2019`](https://philipdarke.com/torchtime/api/data.html#torchtime.data.PhysioNet2019) returns the 2019 challenge (sepsis prediction) [[link]](https://physionet.org/content/challenge-2019/1.0.0/).
* [`torchtime.data.PhysioNet2019Binary`](https://philipdarke.com/torchtime/api/data.html#torchtime.data.PhysioNet2019Binary) returns a binary prediction variant of the 2019 challenge.

For example, to load training data for the 2012 challenge with a 70/30% training/validation split and create a [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for model training:

```python
from torch.utils.data import DataLoader
from torchtime.data import PhysioNet2012

physionet2012 = PhysioNet2012(
    split="train",
    train_prop=0.7,
)
dataloader = DataLoader(physionet2012, batch_size=32)
```

### UEA/UCR repository data sets

The [`torchtime.data.UEA`](https://philipdarke.com/torchtime/api/data.html#torchtime.data.UEA) class returns the [UEA/UCR repository](https://www.timeseriesclassification.com/) data set specified by the `dataset` argument, for example:

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

arrowhead = UEA(
    dataset="ArrowHead",
    split="train",
    train_prop=0.7,
)
dataloader = DataLoader(arrowhead, batch_size=32)
```

### Using the DataLoader

Batches are dictionaries of tensors `X`, `y` and `length`:

* `X` are the time series data. The package follows the *batch first* convention therefore `X` has shape (*n*, *s*, *c*) where *n* is batch size, *s* is (longest) trajectory length and *c* is the number of channels. By default, the first channel is a time stamp.
* `y` are one-hot encoded labels of shape (*n*, *l*) where *l* is the number of classes.
* `length` are the length of each trajectory (before padding if sequences are of irregular length) i.e. a tensor of shape (*n*).

For example, ArrowHead is a univariate time series therefore `X` has two channels, the time stamp followed by the time series (*c* = 2). Each series has 251 observations (*s* = 251) and there are three classes (*l* = 3). For a batch size of 32:

```python
next_batch = next(iter(dataloader))
next_batch["X"].shape       # torch.Size([32, 251, 2])
next_batch["y"].shape       # torch.Size([32, 3])
next_batch["length"].shape  # torch.Size([32])
```

See [Using DataLoaders](https://philipdarke.com/torchtime/tutorials/getting_started.html#using-dataloaders) for more information.

## Advanced options

* Missing data can be imputed by setting `impute` to *mean* (replace with training data channel means) or *forward* (replace with previous observation). Alternatively a custom imputation function can be passed to the `impute` argument.
* A time stamp (added by default), missing data mask and the time since previous observation can be appended with the boolean arguments ``time``, ``mask`` and ``delta`` respectively.
* Time series data are standardised using the `standardise` boolean argument.
* The location of cached data can be changed with the ``path`` argument, for example to share a single cache location across projects.
* For reproducibility, an optional random `seed` can be specified.
* Missing data can be simulated using the `missing` argument to drop data at random from UEA/UCR data sets.

See the [tutorials](https://philipdarke.com/torchtime/tutorials/) and [API](https://philipdarke.com/torchtime/api/) for more information.

## Other resources

If you're looking for the TensorFlow equivalent for PhysioNet data sets try [medical_ts_datasets](https://github.com/ExpectationMax/medical_ts_datasets).

## Acknowledgements

`torchtime` uses some of the data processing ideas in Kidger et al, 2020 [[1]](https://arxiv.org/abs/2005.08926) and Che et al, 2018 [[2]](https://doi.org/10.1038/s41598-018-24271-9).

This work is supported by the Engineering and Physical Sciences Research Council, Centre for Doctoral Training in Cloud Computing for Big Data, Newcastle University (grant number EP/L015358/1).

## References

1. Kidger, P, Morrill, J, Foster, J, *et al*. Neural Controlled Differential Equations for Irregular Time Series. *arXiv* 2005.08926 (2020). [[arXiv]](https://arxiv.org/abs/2005.08926)

1. Che, Z, Purushotham, S, Cho, K, *et al*. Recurrent Neural Networks for Multivariate Time Series with Missing Values. *Sci Rep* 8, 6085 (2018). [[doi]](https://doi.org/10.1038/s41598-018-24271-9)

1. Silva, I, Moody, G, Scott, DJ, *et al*. Predicting In-Hospital Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012. *Comput Cardiol* 2012;39:245-248 (2010). [[hdl]](http://hdl.handle.net/1721.1/93166)

1. Reyna, M, Josef, C, Jeter, R, *et al*. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge. *Critical Care Medicine* 48 2: 210-217 (2019). [[doi]](https://doi.org/10.1097/CCM.0000000000004145)

1. Reyna, M, Josef, C, Jeter, R, *et al*. Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019 (version 1.0.0). *PhysioNet* (2019). [[doi]](https://doi.org/10.13026/v64v-d857)

1. Goldberger, A, Amaral, L, Glass, L, *et al*. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation* 101 (23), pp. e215–e220 (2000). [[doi]](https://doi.org/10.1161/01.cir.101.23.e215)

1. Löning, M, Bagnall, A, Ganesh, S, *et al*. sktime: A Unified Interface for Machine Learning with Time Series. *Workshop on Systems for ML at NeurIPS 2019* (2019). [[doi]](https://doi.org/10.5281/zenodo.3970852)

1. Löning, M, Bagnall, A, Middlehurst, M, *et al*. alan-turing-institute/sktime: v0.10.1 (v0.10.1). *Zenodo* (2022). [[doi]](https://doi.org/10.5281/zenodo.6191159)

## License

Released under the MIT license.
