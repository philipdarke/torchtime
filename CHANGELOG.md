# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2022-03-31

### Added

* Missing data simulation for UEA/UCR data sets
* Support appending missing data masks and time delta channels
* `packed_sequence` collate function
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

[Unreleased]: https://github.com/philipdarke/torchtime/compare/v0.1.1..HEAD
[0.1.1]: https://github.com/philipdarke/torchtime/compare/v0.1.0..v0.1.1
[0.1.0]: https://github.com/philipdarke/torchtime/releases/tag/v0.1.0