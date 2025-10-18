from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import jaxtyping as jt
import torch as th
from torch import nn
from typing_extensions import override

from ml_utils.torch_utils.misc import append_dimensions
from ml_utils.utils import exists

from .base import BaseComponent


@dataclass(frozen=True)
class ResidualConfig:
    """Configuration for residual wrappers.

    Args:
        norm_name: Name of the normalisation layer to use.
            One of "layer", "rms", or None. If None, no normalisation is applied.
        norm_eps: Epsilon value for the normalisation layer.
        use_layer_scale: Whether to use layer scaling.
        layer_scale_init_epsilon: Initial value for layer scaling.
        layer_scale_init_method: Method for initializing layer scaling.
    """

    norm_name: Literal["layer", "rms"] | None = "layer"
    norm_eps: float = 1e-5
    use_layer_scale: bool = True
    layer_scale_init_epsilon: float = 1e-5
    layer_scale_init_method: Literal["constant", "uniform"] = "uniform"


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


def setup_layer_scale(config: ResidualConfig | None, dim: int) -> nn.Parameter | None:
    """Normal setup for layer scaling.

    Args:
        config: Configuration for layer scaling. If None, no layer scaling is applied.
        dim: Dimensionality of the value to be scaled.

    Returns:
        A learnable parameter for layer scaling, or None if no layer scaling is applied.
    """
    if config is None:
        return None
    if config.layer_scale_init_method == "constant":
        return nn.Parameter(config.layer_scale_init_epsilon * th.ones(dim))
    if config.layer_scale_init_method == "uniform":
        return nn.Parameter(th.empty(dim).uniform_(0, 2 * config.layer_scale_init_epsilon))
    raise ValueError(f"Unknown init_method: {config.layer_scale_init_method}")


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

    @override
    @property
    def in_features(self) -> int:
        """Input dimension of this module."""
        return self.wrapped_component.in_features

    @override
    @property
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


class Residual(Wrapper):
    """Wraps a module with a normalisation layer and residual connection.

    Args:
        component (BaseComponent): Wrapped component.
        config (ResidualConfig | None): Configuration for residual wrapper.
    """
    _base_local_attrs: ClassVar[set[str]] = (
        Wrapper._base_local_attrs
        | {"norm", "layer_scale", "_config"}
    )

    def __init__(
        self,
        component: BaseComponent,
        config: ResidualConfig | None = None,
    ) -> None:
        """Initialises the PreNormResidual module.

        Args:
            component: Wrapped component.
            config: Configuration for residual wrapper.
        """
        super().__init__(component)
        config = config if exists(config) else ResidualConfig()
        self._config = config
        if config.norm_name == "layer":
            self.norm = nn.LayerNorm(
                self.in_features, elementwise_affine=False, eps=config.norm_eps
            )
        elif config.norm_name == "rms":
            self.norm = nn.RMSNorm(self.in_features, elementwise_affine=False, eps=config.norm_eps)
        elif not exists(config.norm_name):
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_name: {config.norm_name}")
        self.layer_scale = setup_layer_scale(config, self.out_features)

    def __repr__(self) -> str:
        return f"Residual-{self.wrapped_component}"

    @property
    def has_layer_scale(self) -> bool:
        """Whether layer scaling is applied."""
        return self.layer_scale is not None

    @override
    @property
    def _local_attrs(self) -> set[str]:
        return self._base_local_attrs

    def forward(
        self,
        x: jt.Float[th.Tensor, "*batches dim"],
        *args,
        **kwargs,
    ) -> jt.Float[th.Tensor, "*batches dim"]:
        if self.has_layer_scale:
            return x + self.layer_scale * self.wrapped_component(
                self.norm(x), *args, **kwargs
            )
        return x + self.wrapped_component(self.norm(x), *args, **kwargs)


class ResidualWithContext(Wrapper):
    """Wraps a module with a normalisation layer, residual connection, and gating.

    If context is provided, it is used for adaptive normalisation and gating.
    Gating is always initialised as zero, so the module is initially bypassed.
    """
    _base_local_attrs: ClassVar[set[str]] = (
        Wrapper._base_local_attrs
        | {"_context_dim", "_config", "norm", "scale", "shift", "layer_scale_gate"}
    )

    def __init__(
        self,
        component: BaseComponent,
        context_dim: int,
        config: ResidualConfig | None = None,
    ) -> None:
        """Initialises the ResidualWithContext module.

        Args:
            component: Wrapped component.
            context_dim: Dimension of the context vector.
            config: Configuration for residual wrapper.
        """
        super().__init__(component)
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive: {context_dim}")
        config = config if exists(config) else ResidualConfig()
        self._context_dim = context_dim
        self._config = config
        self._setup_layer_scale(config)

        self.norm = nn.LayerNorm(self.in_features, elementwise_affine=False)
        self.scale = nn.Linear(context_dim, self.out_features)
        self.shift = nn.Linear(context_dim, self.out_features)
        self.reset_parameters()

    def _setup_layer_scale(self, config: ResidualConfig) -> None:
        self.layer_scale_gate = (
            nn.Linear(self._context_dim, self.out_features)
            if config.use_layer_scale
            else nn.Identity()
        )

    @property
    def has_layer_scale(self) -> bool:
        """Whether layer scaling is applied."""
        return self._config.use_layer_scale

    @property
    def context_dim(self) -> int:
        """Dimension of the context vector."""
        return self._context_dim

    @override
    @property
    def _local_attrs(self) -> set[str]:
        return self._base_local_attrs

    def reset_parameters(self) -> None:
        self.scale.weight.data.zero_()
        self.scale.bias.data.zero_()
        self.shift.weight.data.zero_()
        self.shift.bias.data.zero_()
        if self.has_layer_scale:
            assert isinstance(self.layer_scale_gate, nn.Linear)
            if self._config.layer_scale_init_method == "constant":
                self.layer_scale_gate.weight.data.fill_(
                    self._config.layer_scale_init_epsilon
                )
                self.layer_scale_gate.bias.data.fill_(
                    self._config.layer_scale_init_epsilon
                )
            elif self._config.layer_scale_init_method == "uniform":
                self.layer_scale_gate.weight.data.uniform_(
                    0, 2 * self._config.layer_scale_init_epsilon
                )
                self.layer_scale_gate.bias.data.uniform_(
                    0, 2 * self._config.layer_scale_init_epsilon
                )
            else:
                raise ValueError(
                    f"Unknown init_method: {self._config.layer_scale_init_method}"
                )

    def __repr__(self) -> str:
        return f"Context-Residual{self.component}"

    def forward(
        self,
        x: jt.Float[th.Tensor, "*batches dim"],
        *args,
        context: th.Tensor | None = None,
        **kwargs,
    ) -> jt.Float[th.Tensor, "*batches dim"]:
        assert context is not None, f"{self} initialised with ctxt_dim but none given!"
        assert isinstance(self.layer_scale_gate, nn.Linear)
        context = nn.functional.silu(context)
        scale = append_dimensions(self.scale(context), x.dim(), dim=1)
        shift = append_dimensions(self.shift(context), x.dim(), dim=1)
        gate = append_dimensions(self.layer_scale_gate(context), x.dim(), dim=1)
        tmp = self.norm(x) * (scale + 1) + shift
        return x + self.wrapped_component(tmp, *args, **kwargs) * gate


class DropPath(Wrapper):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    Reference: `Deep Networks with Stochastic Depth` - https://arxiv.org/abs/1603.09382.

    Args:
        component (BaseComponent): The component to apply DropPath to.
        drop_prob (float): Probability of an element to be zeroed. Default: 0.0
    """
    _base_local_attrs: ClassVar[set[str]] = Wrapper._base_local_attrs | {"drop_prob"}

    def __init__(self, component: BaseComponent, drop_prob: float = 0.0) -> None:
        """Constructor for DropPath.

        Args:
            component: The component to apply DropPath to.
            drop_prob: Probability of an element to be zeroed. Default: 0.0
        """
        super().__init__(component)
        self.drop_prob = drop_prob

    @override
    def forward(
        self, x: jt.Float[th.Tensor, "*batches dim"], *args, **kwargs
    ) -> jt.Float[th.Tensor, "*batches dim"]:
        """Randomly drops the wrapped component's output during training.

        Args:
            x: Input tensor of shape `(*batches, dim)`
            *args: Additional positional arguments to pass to the wrapped component.
            **kwargs: Additional keyword arguments to pass to the wrapped component.

        Returns:
            Output tensor of shape `(*batches, dim)`

        """
        if self.drop_prob == 0.0 or not self.training:
            return self.wrapped_component(x, *args, **kwargs)
        keep_prob = 1 - self.drop_prob
        random_tensor = th.rand(x.shape[0], device=x.device, dtype=x.dtype)
        dropped = (random_tensor < keep_prob).to(x.dtype)
        dropped = append_dimensions(dropped, x.dim())
        output = self.wrapped_component(x, *args, **kwargs)
        return output.div(keep_prob) * dropped

    @override
    @property
    def _local_attrs(self) -> set[str]:
        return self._base_local_attrs
