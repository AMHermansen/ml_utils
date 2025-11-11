from torch import nn


def reset_parameters(module: nn.Module) -> None:
    """Resets the parameters of the given neural network module.

    Args:
        module (nn.Module): The neural network module whose parameters are to be reset.
    """
    def _apply_reset_parameter(m: nn.Module) -> None:
        if hasattr(m, 'reset_parameters') and callable(m.reset_parameters):
            m.reset_parameters()
    module.apply(_apply_reset_parameter)
