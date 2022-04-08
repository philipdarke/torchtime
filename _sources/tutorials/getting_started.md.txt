# Getting started

This tutorial covers the `torchtime.data.UEA` class for data sets held in the UEA/UCR classification repository [[link]](https://www.timeseriesclassification.com/). A similar approach applies for other data sets, see the [API](../api/data) for further details.

`torchtime.data.UEA` has the following arguments:

* The data set is specified using the `dataset` argument (see list [here](https://www.timeseriesclassification.com/dataset.php)).

* The `split` argument determines whether training, validation or test data are returned. The size of the splits are controlled with the `train_prop` and `val_prop` arguments. See [data splits](data_splits) below.
  
* Missing data can be simulated by dropping data at random. Support is also provided to impute missing data. These options are controlled by the `missing` and `impute` arguments. See the [missing data](missing_data) tutorial for their usage.

* A time stamp, missing data mask and the time since previous observation can be appended to the time series data with the boolean arguments ``time``, ``mask`` and ``delta`` respectively. See the [missing data](missing_data) tutorial for their usage.

* For reproducibility, an optional random `seed` can be specified.

For example, to load training data for the [ArrowHead](https://www.timeseriesclassification.com/description.php?Dataset=ArrowHead) data set with a 70/30% training/validation split:

```python
from torchtime.data import UEA

arrowhead = UEA(
    dataset="ArrowHead",
    split="train",
    train_prop=0.7,
    seed=123,
)
```

`torchtime` downloads the data set using the [`sktime`](https://www.sktime.org) package. Training and validation splits are inconsistent across UEA/UCR data sets therefore the package downloads all data and initialises training, validation and (optional) test data sets.

## Accessing the data

Data can be accessed with the `X`, `y` and `length` attributes. These return the data specified in the `split` argument i.e. the training data in the example above.

* `X` are the time series data. The package follows the *batch first* convention therefore `X` has shape (*n*, *s*, *c*) where *n* is batch size, *s* is (maximum) trajectory length and *c* is the number of channels. By default, a time stamp is appended to the time series data as the first channel.

* `y` are label data. For UEA/UCR data sets, labels are one-hot encoded tensors of shape (*n*, *l*) where *l* is the number of classes.

* `length` are the length of each trajectory (before padding if series are of irregular length) i.e. a tensor of shape (*n*).

Training, validation and test (if specified) data are also available by appending `_train`, `_val` and `_test` respectively.

ArrowHead is a univariate time series therefore `X` has two channels, the time stamp followed by the time series (*c* = 2). Each series has 251 observations (*s* = 251) and there are three classes (*l* = 3).

```python
# Training data (implicit)
arrowhead.X.shape             # torch.Size([148, 251, 2])
arrowhead.y.shape             # torch.Size([148, 3])
arrowhead.length.shape        # torch.Size([148])

# Training data (explicit)
arrowhead.X_train.shape       # torch.Size([148, 251, 2])
arrowhead.y_train.shape       # torch.Size([148, 3])
arrowhead.length_train.shape  # torch.Size([148])

# Validation data
arrowhead.X_val.shape         # torch.Size([63, 251, 2])
arrowhead.y_val.shape         # torch.Size([63, 3])
arrowhead.length_val.shape    # torch.Size([63])
```

## Using DataLoaders

Data sets are passed to a PyTorch [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for model training.

Rather than calling `torchtime.data.UEA` two or three times to create training, validation and test sets, it is more efficient to create one instance of the data set and pass the validation and test data to [`torch.utils.data.TensorDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset). This avoids holding two/three complete copies of the data in memory.

```python
from torch.utils.data import DataLoader, TensorDataset
from torchtime.data import UEA

arrowhead = UEA(
    dataset="ArrowHead",
    split="train",
    train_prop=0.7,
    seed=123,
)
train_dataloader = DataLoader(arrowhead, batch_size=32)

val_data = TensorDataset(
    arrowhead.X_val,
    arrowhead.y_val,
    arrowhead.length_val,
)
val_dataloader = DataLoader(val_data, batch_size=32)
```

In the example above, `train_dataloader` and `val_dataloader` are the iterable DataLoaders for the training and validation data respectively.

DataLoader batches are dictionaries of tensors `X`, `y` and `length`.

(data_splits)=
## Data splits

To create training and validation data sets, pass the proportion of data for the training data to the `train_prop` argument. See the 70/30% training/validation split in the example above.

To create training, validation and test data sets, use both the `train_prop` and `val_prop` arguments. For example, for a 70/20/10% training/validation/test split:

```python
from torchtime.data import UEA

arrowhead = UEA(
    dataset="ArrowHead",
    split="train",
    train_prop=0.7,
    val_prop=0.2,
    seed=123,
)

# Training data (implicit)
arrowhead.X.shape             # torch.Size([148, 251, 2])
arrowhead.y.shape             # torch.Size([148, 3])
arrowhead.length.shape        # torch.Size([148])

# Training data (explicit)
arrowhead.X_train.shape       # torch.Size([148, 251, 2])
arrowhead.y_train.shape       # torch.Size([148, 3])
arrowhead.length_train.shape  # torch.Size([148])

# Validation data
arrowhead.X_val.shape         # torch.Size([42, 251, 2])
arrowhead.y_val.shape         # torch.Size([42, 3])
arrowhead.length_val.shape    # torch.Size([42])

# Test data
arrowhead.X_test.shape        # torch.Size([21, 251, 2])
arrowhead.y_test.shape        # torch.Size([21, 3])
arrowhead.length_test.shape   # torch.Size([21])
```
