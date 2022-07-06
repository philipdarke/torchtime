# Getting started

*This tutorial covers the `torchtime.data.UEA` class. A similar approach applies for other data sets, see the [API](../api/data) for further details.*

`torchtime.data.UEA` has the following arguments:

* The data set is specified using the `dataset` argument (see list [here](https://www.timeseriesclassification.com/dataset.php)).

* The `split` argument determines whether training, validation or test data are returned. The size of the splits are controlled with the `train_prop` and `val_prop` arguments. See [below](data_splits).
  
* Missing data can be simulated by dropping data at random. Support is also provided to impute missing data. These options are controlled by the `missing` and `impute` arguments. See the [missing data](missing_data) tutorial for their usage.

* A time stamp (added by default), missing data mask and the time since previous observation can be appended with the boolean arguments ``time``, ``mask`` and ``delta`` respectively. See the [missing data](missing_data) tutorial for their usage.

* Time series data are standardised using the `standardise` boolean argument (default `False`).

* The location of cached data can be changed with the ``path`` argument (default ``./.torchtime/[dataset name]``), for example to share a single cache location across projects.

* For reproducibility, an optional random `seed` can be specified.

For example, to load training data for the [ArrowHead](https://www.timeseriesclassification.com/description.php?Dataset=ArrowHead) data set with a 70/30% training/validation split as a `torchtime` object named `arrowhead`:

```{eval-rst}
.. testcode::

    from torchtime.data import UEA

    arrowhead = UEA(
        dataset="ArrowHead",
        split="train",
        train_prop=0.7,
        seed=123,  # for reproducibility
    )

.. testoutput::
    :hide:

    ...
```

`torchtime` downloads the data set and extracts the time series using the [`sktime`](https://www.sktime.org) package. Training and validation splits are inconsistent across UEA/UCR data sets therefore the package downloads all data and returns the data splits specified by the `train_prop` and `val_prop` arguments.

## Working with `torchtime` objects

Data are accessed with the `X`, `y` and `length` attributes. These return the data specified in the `split` argument i.e. the training data in the example above.

* `X` are the time series data. The package follows the *batch first* convention therefore `X` has shape (*n*, *s*, *c*) where *n* is batch size, *s* is (longest) trajectory length and *c* is the number of channels. By default, the first channel is a time stamp.

* `y` are one-hot encoded labels of shape (*n*, *l*) where *l* is the number of classes.

* `length` are the length of each trajectory (before padding if sequences are of irregular length) i.e. a tensor of shape (*n*).

Training, validation and test (if specified) data are accessed by appending `_train`, `_val` and `_test` respectively.

ArrowHead is a univariate time series therefore `X` has two channels, the time stamp followed by the time series (*c* = 2). Each series has 251 observations (*s* = 251) and there are three classes (*l* = 3). Therefore, for the example above:

```{eval-rst}
.. doctest::

    >>> # Training data (implicit)
    >>> arrowhead.X.shape
    torch.Size([148, 251, 2])
    >>> arrowhead.y.shape
    torch.Size([148, 3])
    >>> arrowhead.length.shape
    torch.Size([148])

    >>> # Training data (explicit)
    >>> arrowhead.X_train.shape
    torch.Size([148, 251, 2])
    >>> arrowhead.y_train.shape
    torch.Size([148, 3])
    >>> arrowhead.length_train.shape
    torch.Size([148])

    >>> # Validation data
    >>> arrowhead.X_val.shape
    torch.Size([63, 251, 2])
    >>> arrowhead.y_val.shape
    torch.Size([63, 3])
    >>> arrowhead.length_val.shape
    torch.Size([63])
```

(data_splits)=
## Training, validation and test splits

To create training and validation data sets, pass the proportion of data for training to the `train_prop` argument as in the example above.

To create training, validation and test data sets, use both the `train_prop` and `val_prop` arguments. For example, for a 70/20/10% training/validation/test split:

```{eval-rst}
.. testcode::

    arrowhead = UEA(
        dataset="ArrowHead",
        split="train",
        train_prop=0.7,  # 70% training
        val_prop=0.2,    # 20% validation
        seed=123,
    )

.. testoutput::
    :hide:

    ...
```

```{eval-rst}
.. doctest::

    >>> # Training data (implicit)
    >>> arrowhead.X.shape
    torch.Size([148, 251, 2])
    >>> arrowhead.y.shape
    torch.Size([148, 3])
    >>> arrowhead.length.shape
    torch.Size([148])

    >>> # Training data (explicit)
    >>> arrowhead.X_train.shape
    torch.Size([148, 251, 2])
    >>> arrowhead.y_train.shape
    torch.Size([148, 3])
    >>> arrowhead.length_train.shape
    torch.Size([148])

    >>> # Validation data
    >>> arrowhead.X_val.shape
    torch.Size([42, 251, 2])
    >>> arrowhead.y_val.shape
    torch.Size([42, 3])
    >>> arrowhead.length_val.shape
    torch.Size([42])

    >>> # Test data
    >>> arrowhead.X_test.shape
    torch.Size([21, 251, 2])
    >>> arrowhead.y_test.shape
    torch.Size([21, 3])
    >>> arrowhead.length_test.shape
    torch.Size([21])
```

## Using DataLoaders

Data sets are typically passed to a PyTorch [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for model training. Batches of `torchtime` data sets are dictionaries of tensors `X`, `y` and `length`.

Rather than calling `torchtime.data.UEA` two or three times to create training, validation and test sets, it is more efficient to create one instance of the data set and pass the validation and test data to [`torch.utils.data.TensorDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset). This avoids holding two/three complete copies of the data in memory. For example:

```{eval-rst}
.. testcode::

    from torch.utils.data import DataLoader, TensorDataset
    
    arrowhead = UEA(
        dataset="ArrowHead",
        split="train",
        train_prop=0.7,  # 70% training
        val_prop=0.2,    # 20% validation
        seed=123,
    )
    train_dataloader = DataLoader(arrowhead, batch_size=32)

    # Validation data
    val_data = TensorDataset(
        arrowhead.X_val,
        arrowhead.y_val,
        arrowhead.length_val,
    )
    val_dataloader = DataLoader(val_data, batch_size=32)


    # Test data
    test_data = TensorDataset(
        arrowhead.X_test,
        arrowhead.y_test,
        arrowhead.length_test,
    )
    test_dataloader = DataLoader(test_data, batch_size=32)

.. testoutput::
    :hide:

    ...
```

`arrowhead` is a `torchtime` object containing the training and validation data. `train_dataloader`, `val_dataloader` and `test_dataloader` are the iterable DataLoaders for the training, validation and test data respectively.

Note that `train_dataloader` returns batches as a named dictionary as above, but `val_dataloader` and `test_dataloader` return a list.
