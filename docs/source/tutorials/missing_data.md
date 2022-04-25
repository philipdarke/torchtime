# Working with missing data

*This tutorial covers the `torchtime.data.UEA` class for data sets held in the UEA/UCR classification repository [[link]](https://www.timeseriesclassification.com/) however the imputation examples also apply to other data sets.*

## Simulating missing data

Most UEA/UCR data sets are regularly sampled and fully observed. However we often need to work with time series that are irregularly sampled, partially observed and of unequal length. To aid research and model development in this area, missing data can be simulated in UEA/UCR data sets using the `missing` argument.

```{eval-rst}
.. note::
   Data are dropped at random. The ``missing`` argument is the probability that data are missing. Results can be reproduced using the ``seed`` argument.
```

### Regularly sampled data with missing time points

If `missing` is a single value, data are dropped across all channels. This simulates regularly sampled data where some time points are not recorded. Using the [CharacterTrajectories](http://timeseriesclassification.com/description.php?Dataset=CharacterTrajectories) data set as an example:

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_prop=0.7,
    missing=0.5,  # 50% missing
    seed=123,
)
dataloader = DataLoader(char_traj, batch_size=32)
next(iter(dataloader))["X"][0, 0:10]
```

Output:

```python
tensor([[ 0.0000, -0.1849,  0.1978,  0.3263],
        [ 1.0000,     nan,     nan,     nan],
        [ 2.0000, -0.3744,  0.2511,  0.4260],
        [ 3.0000,     nan,     nan,     nan],
        [ 4.0000,     nan,     nan,     nan],
        [ 5.0000,     nan,     nan,     nan],
        [ 6.0000,     nan,     nan,     nan],
        [ 7.0000, -1.0270, -0.1670,  0.2144],
        [ 8.0000,     nan,     nan,     nan],
        [ 9.0000, -1.3501, -0.4994,  0.2447]])
```

### Regularly sampled data with partial observation

Alternatively, data can be dropped independently for each channel by passing a list representing the proportion missing for each channel. This simulates regularly sampled data with partial observation i.e. not all channels are recorded at each time point.

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_prop=0.7,
    missing=[0.8, 0.2, 0.5],  # 80/20/50% missing respectively
    seed=123,
)
dataloader = DataLoader(char_traj, batch_size=32)
next(iter(dataloader))["X"][0, 0:10]
```

Output:

```python
tensor([[ 0.0000,     nan,  0.1978,  0.3263],
        [ 1.0000,     nan,  0.2399,     nan],
        [ 2.0000,     nan,  0.2511,     nan],
        [ 3.0000,     nan,     nan,  0.4016],
        [ 4.0000,     nan,     nan,  0.3410],
        [ 5.0000,     nan,  0.0824,  0.2739],
        [ 6.0000,     nan, -0.0302,  0.2281],
        [ 7.0000,     nan, -0.1670,  0.2144],
        [ 8.0000,     nan,     nan,     nan],
        [ 9.0000, -1.3501, -0.4994,  0.2447]])
```

Note that each time point has a varying number of observations.

## Missing data masks

In some applications, the presence (or absence) of data can itself be informative. For example, a doctor may be more likely to order a particular diagnostic test if they believe the patient has a medical condition. Missing data/observational masks can be used to inform models of missing data. These are appended by setting `mask` to `True`.

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_prop=0.7,
    missing=[0.8, 0.2, 0.5],
    mask=True,
    seed=123,
)
dataloader = DataLoader(char_traj, batch_size=32)
next(iter(dataloader))["X"][0, 0:10]
```

Output:

```python
tensor([[ 0.0000,     nan,  0.1978,  0.3263,  0.0000,  1.0000,  1.0000],
        [ 1.0000,     nan,  0.2399,     nan,  0.0000,  1.0000,  0.0000],
        [ 2.0000,     nan,  0.2511,     nan,  0.0000,  1.0000,  0.0000],
        [ 3.0000,     nan,     nan,  0.4016,  0.0000,  0.0000,  1.0000],
        [ 4.0000,     nan,     nan,  0.3410,  0.0000,  0.0000,  1.0000],
        [ 5.0000,     nan,  0.0824,  0.2739,  0.0000,  1.0000,  1.0000],
        [ 6.0000,     nan, -0.0302,  0.2281,  0.0000,  1.0000,  1.0000],
        [ 7.0000,     nan, -0.1670,  0.2144,  0.0000,  1.0000,  1.0000],
        [ 8.0000,     nan,     nan,     nan,  0.0000,  0.0000,  0.0000],
        [ 9.0000, -1.3501, -0.4994,  0.2447,  1.0000,  1.0000,  1.0000]])
```

Note the final three channels indicate whether data were recorded.

## Time deltas

Some models require the time since the previous observation as an input e.g. GRU-D. This can be added using the `delta` argument. See [Che et al, 2018](https://doi.org/10.1038/s41598-018-24271-9) for implementation details.

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_prop=0.7,
    missing=[0.8, 0.2, 0.5],
    delta=True,
    seed=123,
)
dataloader = DataLoader(char_traj, batch_size=32)
next(iter(dataloader))["X"][0, 0:10]
```

Output:

```python
tensor([[ 0.0000,     nan,  0.1978,  0.3263,  0.0000,  0.0000,  0.0000],
        [ 1.0000,     nan,  0.2399,     nan,  1.0000,  1.0000,  1.0000],
        [ 2.0000,     nan,  0.2511,     nan,  2.0000,  1.0000,  2.0000],
        [ 3.0000,     nan,     nan,  0.4016,  3.0000,  1.0000,  3.0000],
        [ 4.0000,     nan,     nan,  0.3410,  4.0000,  2.0000,  1.0000],
        [ 5.0000,     nan,  0.0824,  0.2739,  5.0000,  3.0000,  1.0000],
        [ 6.0000,     nan, -0.0302,  0.2281,  6.0000,  1.0000,  1.0000],
        [ 7.0000,     nan, -0.1670,  0.2144,  7.0000,  1.0000,  1.0000],
        [ 8.0000,     nan,     nan,     nan,  8.0000,  1.0000,  1.0000],
        [ 9.0000, -1.3501, -0.4994,  0.2447,  9.0000,  2.0000,  2.0000]])
```

Note the second channel is observed at times 2 and 5 therefore the time delta at time 5 is 3 i.e. 3 time units since last observation. Note that time delta is 0 at time 0 by definition.

## Combining output options

The `time`, `mask` and `delta` arguments can be combined as required:

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_prop=0.7,
    missing=[0.8, 0.2, 0.5],
    time=False,
    mask=True,
    delta=True,
    seed=123,
)
dataloader = DataLoader(char_traj, batch_size=32)
next(iter(dataloader))["X"][0, 0:10]
```

Output:

```python
tensor([[    nan,  0.1978,  0.3263,  0.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000],
        [    nan,  0.2399,     nan,  0.0000,  1.0000,  0.0000,  1.0000,  1.0000,  1.0000],
        [    nan,  0.2511,     nan,  0.0000,  1.0000,  0.0000,  2.0000,  1.0000,  2.0000],
        [    nan,     nan,  0.4016,  0.0000,  0.0000,  1.0000,  3.0000,  1.0000,  3.0000],
        [    nan,     nan,  0.3410,  0.0000,  0.0000,  1.0000,  4.0000,  2.0000,  1.0000],
        [    nan,  0.0824,  0.2739,  0.0000,  1.0000,  1.0000,  5.0000,  3.0000,  1.0000],
        [    nan, -0.0302,  0.2281,  0.0000,  1.0000,  1.0000,  6.0000,  1.0000,  1.0000],
        [    nan, -0.1670,  0.2144,  0.0000,  1.0000,  1.0000,  7.0000,  1.0000,  1.0000],
        [    nan,     nan,     nan,  0.0000,  0.0000,  0.0000,  8.0000,  1.0000,  1.0000],
        [-1.3501, -0.4994,  0.2447,  1.0000,  1.0000,  1.0000,  9.0000,  2.0000,  2.0000]])
```

Note the initial time channel is not returned but the missing data and time delta channels are appended to the data.

## Imputing missing data

Missing data can be imputed using the `impute` argument. `torchtime` currently supports mean and forward imputation as well as custom imputation functions.

```{eval-rst}
.. note::
   Imputation has no impact on the missing data mask or time delta channels!
```

### Mean imputation

Under mean imputation, missing data are replaced with the training data channel mean:

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_prop=0.7,
    missing=[0.8, 0.2, 0.5],
    impute="mean",
    seed=123,
)
dataloader = DataLoader(char_traj, batch_size=32)
next(iter(dataloader))["X"][0, 0:10]
```

Output:

```python
tensor([[ 0.0000,  0.1163,  0.1978,  0.3263],
        [ 1.0000,  0.1163,  0.2399, -0.2935],
        [ 2.0000,  0.1163,  0.2511, -0.2935],
        [ 3.0000,  0.1163, -0.0722,  0.4016],
        [ 4.0000,  0.1163, -0.0722,  0.3410],
        [ 5.0000,  0.1163,  0.0824,  0.2739],
        [ 6.0000,  0.1163, -0.0302,  0.2281],
        [ 7.0000,  0.1163, -0.1670,  0.2144],
        [ 8.0000,  0.1163, -0.0722, -0.2935],
        [ 9.0000, -1.3501, -0.4994,  0.2447]])
```

### Forward imputation

Under forward imputation, missing values are replaced with the previous channel observation. Note that this approach does not impute any initial missing values, therefore these are replaced with the training data channel mean.

This approach ensures that knowledge of the time series at times *t > i* is not used when imputing values at time *i*. This is required when developing models that make online predictions.

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_prop=0.7,
    missing=[0.8, 0.2, 0.5],
    impute="forward",
    seed=123,
)
dataloader = DataLoader(char_traj, batch_size=32)
next(iter(dataloader))["X"][0, 0:10]
```

Output:

```python
tensor([[ 0.0000,  0.1163,  0.1978,  0.3263],
        [ 1.0000,  0.1163,  0.2399,  0.3263],
        [ 2.0000,  0.1163,  0.2511,  0.3263],
        [ 3.0000,  0.1163,  0.2511,  0.4016],
        [ 4.0000,  0.1163,  0.2511,  0.3410],
        [ 5.0000,  0.1163,  0.0824,  0.2739],
        [ 6.0000,  0.1163, -0.0302,  0.2281],
        [ 7.0000,  0.1163, -0.1670,  0.2144],
        [ 8.0000,  0.1163, -0.1670,  0.2144],
        [ 9.0000, -1.3501, -0.4994,  0.2447]])
```

```{eval-rst}
.. note::
   ``torchtime.impute`` includes imputation functions for tensors with missing data. See the `API <../api/impute.html>`_ for more information.
```

### Custom imputation functions

Alternatively a custom imputation function can be passed to ``impute``. This must accept ``X`` and ``y`` tensors with the raw time series and labels respectively and return both tensors post imputation.

```python
from torch.utils.data import DataLoader
from torchtime.data import UEA

def zero_imputation(X, y):
    """Set missing values to 0."""
    return X.nan_to_num(0), y.nan_to_num(0)

char_traj = UEA(
    dataset="CharacterTrajectories",
    split="train",
    train_prop=0.7,
    missing=[0.8, 0.2, 0.5],
    impute=zero_imputation,
    seed=123,
)
dataloader = DataLoader(char_traj, batch_size=32)
next(iter(dataloader))["X"][0, 0:10]
```

Output:

```python
tensor([[ 0.0000,  0.0000,  0.1978,  0.3263],
        [ 1.0000,  0.0000,  0.2399,  0.0000],
        [ 2.0000,  0.0000,  0.2511,  0.0000],
        [ 3.0000,  0.0000,  0.0000,  0.4016],
        [ 4.0000,  0.0000,  0.0000,  0.3410],
        [ 5.0000,  0.0000,  0.0824,  0.2739],
        [ 6.0000,  0.0000, -0.0302,  0.2281],
        [ 7.0000,  0.0000, -0.1670,  0.2144],
        [ 8.0000,  0.0000,  0.0000,  0.0000],
        [ 9.0000, -1.3501, -0.4994,  0.2447]])
```
