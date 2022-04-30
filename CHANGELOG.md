# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* PhysioNet2019Binary data set, a implified binary prediction variant of PhysioNet2019

### Changed

* Updated documentation

## [0.3.0] - 2022-04-25

### Added

* PhysioNet2012 data set
* PhysioNet and UEA/UCR unit tests
* Utility function module
* Better console messages

### Changed

* More efficient PhysioNet data set downscaling
* Updated documentation

### Fixed

* Replace PhysioNet2019 missing data indicator with `NaNs`
* Code coverage badge

## [0.2.0] - 2022-04-08

### Added

* `impute` argument to support missing data imputation using *mean* and *forward* imputation methods or a custom imputation function
* ``downscale`` argument to reduce the size of data sets for testing/model development
* `torchtime.data.TensorTimeSeriesDataset` class to create a data set from input tensors

### Changed

* Processed data are now cached in the ``.torchtime`` directory
* `train_split` and `val_split` arguments are renamed `train_prop` and `val_prop` respectively
* Introduced generic `torchtime.data_TimeSeriesDataSet` class behind the scenes - note training/validation/test data splits have changed for a given seed
* `torchtime.collate.packed_sequence` now returns both `X` and `y` as a PackedSequence object
* Expanded unit tests - note coverage is currently limited as PhysioNet2019 tests cannot be run under CI
* Updated documentation

### Fixed

* Use `float32`/`torch.float` and `int64`/`torch.long` precision for all data sets
* Shape of `y` data in PhysioNet2019 data
* Bug when adding time delta channels without a missing data mask

## [0.1.1] - 2022-03-31

### Added

* Missing data simulation for UEA/UCR data sets
* Support appending missing data masks and time delta channels
* `torchtime.collate.packed_sequence` collate function
* Documentation now includes a tutorial
* Automated releases using GitHub Actions
* DOI

### Changed

* Simplified training/validation/test split approach
* Default file path for PhysioNet2019 data set is now `data/physionet2019`
* Refactored `torchtime.data` to share utility functions across data classes
* Expanded unit tests
* Updated documentation

## [0.1.0] - 2022-03-28

First release to PyPi

[Unreleased]: https://github.com/philipdarke/torchtime/compare/v0.3.0.HEAD
[0.3.0]: https://github.com/philipdarke/torchtime/compare/v0.1.0..v0.3.0
[0.2.0]: https://github.com/philipdarke/torchtime/compare/v0.1.0..v0.2.0
[0.1.1]: https://github.com/philipdarke/torchtime/compare/v0.1.0..v0.1.1
[0.1.0]: https://github.com/philipdarke/torchtime/releases/tag/v0.1.0