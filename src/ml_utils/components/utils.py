from typing import Literal

from torch import nn


def reset_parameters(module: nn.Module) -> None:
    """Resets the parameters of the given neural network module.

    Args:
        module (nn.Module): The neural network module whose parameters are to be reset.
    """

    def _apply_reset_parameter(m: nn.Module) -> None:
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()

    module.apply(_apply_reset_parameter)


def instantiate_norm_layer(
    norm_type: Literal["layer", "rms"] | None,
    dim: int,
) -> nn.Module:
    """Utility function to instantiate a normalization layer based.

    Args:
        norm_type: The type of normalization layer to instantiate.
            Can be "layer" for LayerNorm, "rms" for RMSNorm,
            or None for no normalization.
        dim: The dimension of the input features for the normalization layer.

    Returns:
        An instance of the specified normalization layer or nn.Identity if no
        normalization is specified.

    """
    if norm_type == "layer":
        return nn.LayerNorm(dim)
    if norm_type == "rms":
        return nn.RMSNorm(dim)
    return nn.Identity()
