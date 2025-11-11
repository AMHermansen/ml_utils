import logging
from abc import ABC, abstractmethod

from torch.nn import Module

logger = logging.getLogger(__name__)


class BaseComponent(Module, ABC):

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Number of output features of the component."""

    @property
    @abstractmethod
    def in_features(self) -> int | None:
        """Number of input features of the component. If None, component is agnostic."""

    # Subclasses can optionally implement this method, to define custom parameter reset
    # behavior.
    # We include this method here, to specify what function modules should look for.
    # If this method is called on a subclass that does not implement it, a warning
    # will be logged.
    def reset_parameters(self) -> None:
        """Reset the parameters of the component to their initial state."""
        logger.warning(
            f"{self.__class__.__name__} does not implement reset_parameters(). "
            f"Parameters have not been reset."
        )
