import torch as th


def append_dimensions(x: th.Tensor, target_dimensions: int, dim: int = -1) -> th.Tensor:
    """Appends dimensions of size 1 to a tensor until it reaches the target number of dimensions.

    Args:
        x (th.Tensor): The input tensor.
        target_dimensions (int): The target number of dimensions.
        dim (int, optional): The dimension along which to append. Defaults to -1.

    Returns:
        th.Tensor: The reshaped tensor with the target number of dimensions.
    """
    if (dim_diff := target_dimensions - x.ndim) < 0:
        raise ValueError(
            f"Target dimensions {target_dimensions} must be greater than or equal to current dimensions {x.ndim}."
        )

    # Fast path for common cases
    if dim_diff == 0:
        return x
    if dim_diff == 1:
        return x.unsqueeze(dim)
    if dim == -1:
        return x[(...,) + (None,) * dim_diff]
    if dim == 0:
        return x[(None,) * dim_diff + (...,)]

    allow_range = [-x.dim() - 1, x.dim()]
    if not allow_range[0] <= dim <= allow_range[1]:
        raise ValueError(f"Dimension {dim} out of range {allow_range}.")

    if dim < 0:
        dim += x.dim() + 1
    return x.view(*x.shape[:dim], *dim_diff * (1,), *x.shape[dim:])
