from typing import List

import torch
from torch import Tensor


def replace_missing(input: Tensor, fill: Tensor, select: List = None) -> Tensor:
    """**Replace missing data with a fixed value by channel.**

    Imputes missing data by replacing all ``NaNs`` with a value by channel. All channels
    are imputed by default, however a subset can be imputed by passing a list of indices
    to ``select``.

    ``fill`` must be a tensor with the replacement value for, and in the same order as,
    each channel that is imputed. If all columns are to be imputed, ``fill`` must be a
    tensor of size (*c*) where *c* is the number of channels in the input. If ``select``
    is used to impute a subset of channels, ``fill`` must be a tensor of size (*d*)
    where *d* is the length of ``select``. A common choice of ``fill`` is the mean of
    each channel in the training data.

    Under this approach, no knowledge of the time series at times *t > i* is required
    when imputing values at time *i*. This is essential if you are developing a model
    that will make online predictions.

    Args:
        input: The tensor to impute.
        fill: Fill values for each channel, for example channel means.
        select: Impute these channels (by default all channels are imputed).

    Returns:
        Tensor: Tensor with imputed data.
    """
    assert type(fill) is Tensor, "argument 'fill' must be a Tensor"
    if select is None:
        assert fill.size(0) == input.size(
            -1
        ), "tensor 'fill' must have same number of channels as input ({})".format(
            input.size(-1)
        )
        select = list(torch.arange(input.size(-1)).numpy())
    else:
        assert type(select) is list and fill.size(0) == len(
            select
        ), "'select' must be a list the same length as 'fill' ({})".format(fill.size(0))
    output = input.clone()
    for i, channel in enumerate(torch.unbind(output, dim=-1)):
        if i in select:
            channel.nan_to_num_(fill[0])
            fill = fill[1:]
    return output


def forward_impute(input: Tensor, fill: Tensor = None) -> Tensor:
    """**Imputes missing data using (last observation carried) forward imputation.**

    Missing data (``NaNs``) are replaced by the previous channel observation.

    If the initial value(s) of a channel is ``NaN`` this is replaced with the respective
    value in ``fill``. ``fill`` must be a tensor with the replacement value for, and in
    the same order as, each channel. A common choice of ``fill`` is the mean of each
    channel in the training data.  ``fill` is only required if an initial value is
    ``NaN``.

    Under this approach, no knowledge of the time series at times *t > i* is required
    when imputing values at time *i*. This is essential if you are developing a model
    that will make online predictions.

    .. note::
        Only ``input`` tensors with 3 or fewer dimensions are currently supported. The
        final dimension must hold channel data.

    Args:
        input: The tensor to impute.
        fill: Fill values for initial NaNs in each channel, for example channel means.

    Returns:
        Tensor: Tensor with imputed data.
    """
    assert len(input.size()) <= 3, "tensor 'input' must have 3 or fewer dimensions"
    # Last observation carried forward
    x = input.transpose(-2, -1)  # shape (n, c, s)
    x_mask = torch.logical_not(torch.isnan(x))
    x_mask = torch.cummax(x_mask, -1)[1]
    output = x.gather(-1, x_mask)
    output = output.transpose(-2, -1)  # shape (n, s, c)
    # Fill initial NaNs
    if torch.sum(torch.isnan(output)) > 0:
        assert fill is not None, "argument 'fill' must be provided"
        output = replace_missing(output, fill)
    return output
