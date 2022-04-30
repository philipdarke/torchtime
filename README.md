# Time series data sets for PyTorch

[![PyPi](https://img.shields.io/pypi/v/torchtime)](https://pypi.org/project/torchtime)
[![Build status](https://img.shields.io/github/workflow/status/philipdarke/torchtime/build.svg)](https://github.com/philipdarke/torchtime/actions/workflows/build.yml)
![Coverage](https://philipdarke.com/torchtime/assets/coverage-badge.svg?dummy=8484744)
[![License](https://img.shields.io/github/license/philipdarke/torchtime.svg)](https://github.com/philipdarke/torchtime/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/475093888.svg)](https://zenodo.org/badge/latestdoi/475093888)

Benchmark PyTorch data sets for supervised time series classification and prediction problems. `torchtime` currently supports:

* All data sets in the UEA/UCR classification repository [[link]](https://www.timeseriesclassification.com/)

* PhysioNet Challenge 2012 (in-hospital mortality) [[link]](https://physionet.org/content/challenge-2012/1.0.0/)

* PhysioNet Challenge 2019 (sepsis prediction) [[link]](https://physionet.org/content/challenge-2019/1.0.0/)

## Installation

```bash
$ pip install torchtime
```

## Example usage

`torchtime.data` contains a class for each data set above. Each class has a consistent API.

The `torchtime.data.UEA` class returns the UEA/UCR data set specified by the `dataset` argument (see list of data sets [here](https://www.timeseriesclassification.com/dataset.php)). For example, to load training data for the [ArrowHead](https://www.timeseriesclassification.com/description.php?Dataset=ArrowHead) data set with a 70/30% training/validation split and create a [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):

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

Batches are dictionaries of tensors `X`, `y` and `length`.

`X` are the time series data. The package follows the *batch first* convention therefore `X` has shape (*n*, *s*, *c*) where *n* is batch size, *s* is (maximum) trajectory length and *c* is the number of channels. By default, a time stamp is appended to the time series data as the first channel.

`y` are one-hot encoded labels of shape (*n*, *l*) where *l* is the number of classes and `length` are the length of each trajectory (before padding if sequences are of irregular length) i.e. a tensor of shape (*n*).

ArrowHead is a univariate time series therefore `X` has two channels, the time stamp followed by the time series (*c* = 2). Each series has 251 observations (*s* = 251) and there are three classes (*l* = 3).

```python
next_batch = next(iter(dataloader))

next_batch["X"].shape       # torch.Size([32, 251, 2])
next_batch["y"].shape       # torch.Size([32, 3])
next_batch["length"].shape  # torch.Size([32])
```

## Additional options

* The `split` argument determines whether training, validation or test data are returned. The size of the splits are controlled with the `train_prop` and `val_prop` arguments.

* Missing data can be imputed by setting `impute` to *mean* (replace with training data channel means) or *forward* (replace with previous observation). Alternatively a custom imputation function can be used.

* A time stamp (added by default), missing data mask and the time since previous observation can be appended with the boolean arguments ``time``, ``mask`` and ``delta`` respectively.

* For reproducibility, an optional random `seed` can be specified.

Missing data can be simulated using the `missing` argument to drop data at random from UEA/UCR data sets. See the [tutorials](https://philipdarke.com/torchtime/tutorials/) and [API](https://philipdarke.com/torchtime/api/) for more information.

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
