# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.1] - 2023-06-13

### Fixed

* Support for PyTorch 1.13.x to 2.0.x
* GitHub Actions badge

## [0.6.0] - 2023-06-12

### Added

* Support for Python 3.10 and 3.11

### Changed

* Updated dependencies

## [0.5.1] - 2022-08-03

### Changed

* Adopt updated `sktime` version

## [0.5.0] - 2022-08-01

### Added

* "zero" value imputation
* Download progress bars

### Changed

* One-hot encoded PhysioNet2012 `ICUType` channel
* Updated pre-commit hooks

### Fixed

* `overwrite_cache` now re-downloads data
* `scikit-learn` dependency

## [0.4.2] - 2022-07-06

### Added

* `torchtime.data.PhysioNet2019Binary` data set, a binary prediction variant of the PhysioNet 2019 challenge
* SHA256 checksums to verify integrity of cached data
* `standardise` argument to standarise data
* `overwrite_cache` argument to update a cached data set
* Impute a subset of channels using forward imputation with `select` argument
* Progress bars for PhysioNet data set processing
* Additional argument validation
* Additional unit tests
* Code copy button in documentation

### Changed

* Code refactor
* Removed `torchtime.data.TensorTimeSeriesDataset` class
* Removed `downscale` argument
* `override` argument renamed `channel_means`
* Download UEA/UCR data directly (not via `sktime`)
* `overwrite_cache` argument to update cached data
* Updated console messages
* Rename cache directories
* `test` directory renamed `tests`
* Using MacOS runner for GitHub Actions
* Updated tutorials with automated code testing
* Updated documentation

### Fixed

* Continuous deployment

## [0.4.1] - 2022-07-06

Release pulled

## [0.4.0] - 2022-07-06

Release pulled

## [0.3.0] - 2022-04-25

### Added

* `torchtime.data.PhysioNet2012` data set
* PhysioNet and UEA/UCR unit tests
* Utility function module
* Better console messages

### Changed

* More efficient PhysioNet data set downscaling
* Updated documentation

### Fixed

* Replace PhysioNet 2019 missing data indicator with `NaN`
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

[Unreleased]: https://github.com/philipdarke/torchtime/compare/v0.6.1.HEAD
[0.6.1]: https://github.com/philipdarke/torchtime/compare/v0.6.0..v0.6.1
[0.6.0]: https://github.com/philipdarke/torchtime/compare/v0.5.1..v0.6.0
[0.5.1]: https://github.com/philipdarke/torchtime/compare/v0.5.0..v0.5.1
[0.5.0]: https://github.com/philipdarke/torchtime/compare/v0.4.2..v0.5.0
[0.4.2]: https://github.com/philipdarke/torchtime/compare/v0.3.0..v0.4.2
[0.3.0]: https://github.com/philipdarke/torchtime/compare/v0.2.0..v0.3.0
[0.2.0]: https://github.com/philipdarke/torchtime/compare/v0.1.1..v0.2.0
[0.1.1]: https://github.com/philipdarke/torchtime/compare/v0.1.0..v0.1.1
[0.1.0]: https://github.com/philipdarke/torchtime/releases/tag/v0.1.0
