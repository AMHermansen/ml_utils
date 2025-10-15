from abc import ABC, abstractmethod

from torch.nn import Module


class BaseComponent(Module, ABC):

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Number of output features of the component."""

    @property
    @abstractmethod
    def in_features(self) -> int | None:
        """Number of input features of the component. If None, component is agnostic."""
