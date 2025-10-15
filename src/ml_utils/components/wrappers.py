from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

import jaxtyping as jt
import torch as th
from torch import nn
from typing_extensions import override

from ml_utils.torch_utils.misc import append_dimensions

from .base import BaseComponent


@dataclass(frozen=True)
class LayerScaleConfig:
    init_epsilon_value: float = 1e-5
    init_method: Literal["constant", "uniform"] = "constant"


_SENTINEL_LAYERSCALE_CONFIG = LayerScaleConfig()


def check_feature_preserving(component: BaseComponent):
    """Check if a component is feature-preserving.

    Args:
        component: A component to check.

    Raises:
        ValueError: If the component is not shape-preserving.
    """
    if component.in_dim != component.out_dim:
        raise ValueError(
            f"Component {component} is not an endomorphism: in_dim={component.in_dim}, out_dim={component.out_dim}"
        )


def setup_layer_scale(config: LayerScaleConfig | None, dim: int) -> nn.Parameter | None:
    """Normal setup for layer scaling.

    Args:
        config: Configuration for layer scaling. If None, no layer scaling is applied.
        dim: Dimensionality of the value to be scaled.

    Returns:
        A learnable parameter for layer scaling, or None if no layer scaling is applied.
    """
    if config is None:
        return None
    if config.init_method == "constant":
        return nn.Parameter(config.init_epsilon_value * th.ones(dim))
    if config.init_method == "uniform":
        return nn.Parameter(th.empty(dim).uniform_(0, 2 * config.init_epsilon_value))
    raise ValueError(f"Unknown init_method: {config.init_method}")


class Wrapper(BaseComponent):
    """Base class for wrappers around components.

    Args:
        component (BaseComponent): A component with matching input and output dimensions.
    """

    def __init__(
        self,
        component: BaseComponent,
    ) -> None:
        """Constructor.

        Args:
            component: A component with matching input and output dimensions.
        """
        super().__init__()
        check_feature_preserving(component)
        self.wrapped_component = component

    @override
    @property
    def in_dim(self) -> int:
        """Input dimension of this module."""
        return self.wrapped_component.in_dim

    @override
    @property
    def out_dim(self) -> int:
        """Input dimension of this module."""
        return self.wrapped_component.out_dim

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


class Residual(Wrapper):
    """Wraps a module with a normalisation layer and residual connection.

    One should generally prefer the PreNormResidual wrapper instead.

    Args:
        component (BaseComponent): The component to wrap.
        layer_scale_config (LayerScaleConfig | None): Configuration for layer scaling.
            If None, no layer scaling is applied.
    """

    def __init__(
        self,
        component: BaseComponent,
        layer_scale_config: LayerScaleConfig | None = _SENTINEL_LAYERSCALE_CONFIG,
    ) -> None:
        """Initialises the Residual module.

        Args:
            component: Wrapped component.
            layer_scale_config: Configuration for layer scaling. If None, no layer scaling is applied.
        """
        super().__init__(component)
        self._layer_scale_config = layer_scale_config
        self.layer_scale = setup_layer_scale(layer_scale_config, self.out_dim)

    def forward(
        self,
        x: jt.Float[th.Tensor, "*batches dim"],
        *args,
        **kwargs,
    ) -> jt.Float[th.Tensor, "*batches dim"]:
        if self.has_layer_scale:
            return x + self.layer_scale * self.wrapped_component(x, *args, **kwargs)
        return x + self.wrapped_component(x, *args, **kwargs)

    @property
    def has_layer_scale(self) -> bool:
        """Whether layer scaling is applied."""
        return self.layer_scale is not None

    def __repr__(self) -> str:
        return f"Residual-{self.wrapped_component}"


class PreNormResidual(Wrapper):
    """Wraps a module with a normalisation layer and residual connection."""

    def __init__(
        self,
        component: BaseComponent,
        norm_name: Literal["layer", "rms"],
        eps: float = 1e-5,
        layer_scale_config: LayerScaleConfig | None = _SENTINEL_LAYERSCALE_CONFIG,
    ) -> None:
        """Initialises the PreNormResidual module.

        Args:
            component: Wrapped component.
            norm_name: Type of normalization to use ("layer" for LayerNorm, "rms" for RMSNorm).
            eps: Epsilon value for numerical stability in normalization.
            layer_scale_config: Configuration for layer scaling. If None, no layer scaling is applied.
        """
        super().__init__(component)
        self._layer_scale_config = layer_scale_config
        if norm_name == "layer":
            self.norm = nn.LayerNorm(self.in_dim, elementwise_affine=False, eps=eps)
        elif norm_name == "rms":
            self.norm = nn.RMSNorm(self.in_dim, elementwise_affine=False, eps=eps)
        else:
            raise ValueError(f"Unknown norm_name: {norm_name}")
        self.layer_scale = setup_layer_scale(layer_scale_config, self.out_dim)

    def __repr__(self) -> str:
        return f"Residual-{self.wrapped_component}"

    @property
    def has_layer_scale(self) -> bool:
        """Whether layer scaling is applied."""
        return self.layer_scale is not None

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

    def __init__(
        self,
        component: BaseComponent,
        context_dim: int,
        layer_scale_config: LayerScaleConfig | None = _SENTINEL_LAYERSCALE_CONFIG,
    ) -> None:
        """Initialises the ResidualWithContext module.

        Args:
            component: Wrapped component.
            context_dim: Dimension of the context vector.
            layer_scale_config: Configuration for layer scaling. If None, no layer
                scaling is applied.
        """
        super().__init__(component)
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive: {context_dim}")
        self._context_dim = context_dim
        self._layer_scale_config = layer_scale_config
        self._setup_layer_scale(layer_scale_config)

        self.norm = nn.LayerNorm(self.in_dim, elementwise_affine=False)
        self.scale = nn.Linear(context_dim, self.out_dim)
        self.shift = nn.Linear(context_dim, self.out_dim)
        self.reset_parameters()

    def _setup_layer_scale(self, layer_scale_config: LayerScaleConfig | None) -> None:
        self.layer_scale_gate = (
            nn.Linear(self._context_dim, self.out_dim)
            if layer_scale_config
            else nn.Identity()
        )

    @property
    def has_layer_scale(self) -> bool:
        """Whether layer scaling is applied."""
        return self._layer_scale_config is not None

    @property
    def context_dim(self) -> int:
        """Dimension of the context vector."""
        return self._context_dim

    def reset_parameters(self) -> None:
        self.scale.weight.data.zero_()
        self.scale.bias.data.zero_()
        self.shift.weight.data.zero_()
        self.shift.bias.data.zero_()
        if self.has_layer_scale:
            assert isinstance(self.layer_scale_gate, nn.Linear)
            if self._layer_scale_config.init_method == "constant":
                self.layer_scale_gate.weight.data.zero_()
                self.layer_scale_gate.bias.data.fill_(
                    self._layer_scale_config.init_epsilon_value
                )
            elif self._layer_scale_config.init_method == "uniform":
                self.layer_scale_gate.weight.data.uniform_(
                    0, self._layer_scale_config.init_epsilon_value
                )
                self.layer_scale_gate.bias.data.uniform_(
                    0, self._layer_scale_config.init_epsilon_value
                )
            else:
                raise ValueError(
                    f"Unknown init_method: {self._layer_scale_config.init_method}"
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
