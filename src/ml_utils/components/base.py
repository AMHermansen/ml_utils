from abc import ABC, abstractmethod

from torch.nn import Module


class BaseComponent(Module, ABC):

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Output dimension of the component."""

    @property
    @abstractmethod
    def in_dim(self) -> int | None:
        """Input dimension of the component. Can be None if input dimension is not fixed."""
