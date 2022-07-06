"""
=======================
Missing data imputation
=======================

The following functions are provided to impute missing time series data:

* `Replace missing data with a fixed value <#torchtime.impute.replace_missing>`_
* `Forward imputation <#torchtime.impute.forward_impute>`_
"""

import torch
from torch import Tensor


def replace_missing(input: Tensor, fill: Tensor, select: Tensor = None) -> Tensor:
    """**Replace missing data with a fixed value by channel.**

    Imputes missing data by replacing all ``NaNs`` with a fixed value by channel. Fill
    values are specified by the ``fill`` argument. All channels are imputed by default,
    however a subset can be imputed by passing the indices to ``select``.

    A common choice of ``fill`` is the mean of each channel in the training data. Under
    this approach, no knowledge of the time series at times *t > i* is required when
    imputing values at time *i*. This is essential if you are developing a model that
    will make online predictions.

    Args:
        input: The tensor to impute. The final dimension must hold channel data.
        fill: Fill values for each channel in the same order as the data. ``fill`` must
            be the same length as the number of channels to be imputed i.e. the number
            of channels in the data or the length of ``select`` if shorter.
        select: Indices for the channels to be imputed (by default all channels are
            imputed).

    Returns:
        Imputed time series.
    """
    assert type(fill) is Tensor, "argument 'fill' must be a Tensor"
    if select is None:
        assert fill.size(0) == input.size(
            -1
        ), "Tensor 'fill' must have same number of channels as input ({})".format(
            input.size(-1)
        )
        select = torch.arange(input.size(-1))
    else:
        assert type(select) is Tensor and fill.size(0) == len(
            select
        ), "'select' must be a Tensor the same length as 'fill' ({})".format(
            fill.size(0)
        )
    # Replace missing values by channel
    output = input.clone()
    j = 0
    for i, channel in enumerate(torch.unbind(output, dim=-1)):
        if i in select:
            channel.nan_to_num_(fill[j])
            j += 1
    return output


def forward_impute(input: Tensor, fill: Tensor = None, select: Tensor = None) -> Tensor:
    """**Replace missing data with last observation carried forward.**

    Missing data (``NaNs``) are replaced by the previous observation in the channel.

    If the initial value(s) of a channel is ``NaN`` this is replaced with the respective
    value in ``fill`` (only required if an initial value is ``NaN``). All channels are
    imputed by default, however a subset can be imputed by passing the indices to
    ``select``.

    A common choice of ``fill`` is the mean of each channel in the training data. Under
    this approach, no knowledge of the time series at times *t > i* is required when
    imputing values at time *i*. This is essential if you are developing a model that
    will make online predictions.

    .. note::
        Only ``input`` tensors with 3 or fewer dimensions are currently supported. The
        final dimension must hold channel data.

    Args:
        input: The tensor to impute. The final dimension must hold channel data.
        fill: Fill values for each channel in the same order as the data. ``fill`` must
            be the same length as the number of channels to be imputed i.e. the number
            of channels in the data or the length of ``select`` if shorter.
        select: Indices for the channels to be imputed (by default all channels are
            imputed).

    Returns:
        Imputed time series.
    """
    assert len(input.size()) >= 2, "Tensor 'input' must have at least two dimensions"
    if select is None:
        select = torch.arange(input.size(-1))
    # Last observation carried forward (all channels)
    x = input.transpose(-2, -1)  # shape (n, c, s)
    x_mask = torch.logical_not(torch.isnan(x))
    x_mask = torch.cummax(x_mask, -1)[1]
    x_imputed = x.gather(-1, x_mask)
    x_imputed = x_imputed.transpose(-2, -1)  # shape (n, s, c)
    # Update selected channels with imputed data
    output = input.index_copy(-1, select, x_imputed[..., select])
    # Fill initial NaNs
    if torch.sum(torch.isnan(output[..., select])) > 0:
        assert fill is not None, "argument 'fill' must be provided"
        output = replace_missing(output, fill, select)
    return output
