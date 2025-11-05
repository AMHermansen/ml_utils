from abc import abstractmethod
from typing import Any, ClassVar, cast

import jaxtyping as jt
import torch as th
from typing_extensions import override

from ml_utils.components.base import BaseComponent


def check_feature_preserving(component: BaseComponent):
    """Check if a component is feature-preserving.

    Args:
        component: A component to check.

    Raises:
        ValueError: If the component is not shape-preserving.
    """
    if component.in_features != component.out_features:
        raise ValueError(
            f"Component {component} is not feature-preserving: in_dim={component.in_features}, out_dim={component.out_features}"
        )


class Wrapper(BaseComponent):
    """Base class for wrappers around components.

    When this is subclassed, it is very important to include all local attributes
    in the `_base_local_attrs` set to avoid infinite recursion in `__getattr__
    and `__setattr__`.

    Args:
        wrapped_component (BaseComponent): A component with matching input and output
            dimensions.
    """

    # It is EXTREMELY important that all local attributes are included here to avoid
    # infinite recursion in __getattr__ and __setattr__.
    _base_local_attrs: ClassVar[set[str]] = {"wrapped_component"}

    def __init__(
        self,
        wrapped_component: BaseComponent,
    ) -> None:
        """Constructor.

        Args:
            wrapped_component: A component with matching input and output dimensions.
        """
        super().__init__()
        check_feature_preserving(wrapped_component)
        self.wrapped_component = wrapped_component

    @property
    @override
    def in_features(self) -> int:
        """Input dimension of this module."""
        assert isinstance(self.wrapped_component.in_features, int), "Expected in_features to be an int"
        return self.wrapped_component.in_features

    @property
    @override
    def out_features(self) -> int:
        """Input dimension of this module."""
        return self.wrapped_component.out_features

    # We're making this abstract to get additional confirmation that subclasses
    # are including their local attributes in _base_local_attrs.
    @property
    @abstractmethod
    def _local_attrs(self) -> set[str]:
        return self._base_local_attrs

    @abstractmethod
    def forward(
        self,
        x: jt.Float[th.Tensor, "*batches dim"],
        *args,
        **kwargs,
    ) -> jt.Float[th.Tensor, "*batches dim"]:
        """Applies a wrapping effect around the wrapped component.

        Args:
            x: input tensor of shape `(*batches, dim)`
            *args: additional positional arguments to pass to the wrapped component
            **kwargs: additional keyword arguments to pass to the wrapped component

        Returns:
            Output tensor of shape `(*batches, dim)`
        """

    # Moderately cursed python, to delegate attribute access to the wrapped components.
    # This makes it much cleaner to interact with the wrapped component's attributes.
    def __getattr__(self, item: str) -> Any:
        """Delegate attribute access to the wrapped component if not found.

        Args:
            item: Attribute name.

        Returns:
            Attribute value.
        """
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.wrapped_component, item)

    def __setattr__(self, key: str, value: Any) -> None:
        """Delegate attribute setting to the wrapped component if not found."""
        local_attrs = self._local_attrs

        if key in local_attrs or not hasattr(self, "wrapped_component"):
            super().__setattr__(key, value)
        else:
            setattr(self.wrapped_component, key, value)
