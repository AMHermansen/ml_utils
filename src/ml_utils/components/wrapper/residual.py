from dataclasses import dataclass
from typing import ClassVar, Literal

import jaxtyping as jt
import torch as th
from torch import nn
from typing_extensions import override

from ml_utils.components.base import BaseComponent
from ml_utils.torch_utils.misc import append_dimensions
from ml_utils.utils import exists

from .base import Wrapper
from .drop import (
    maybe_apply_dropout,
    maybe_apply_stochastic_depth,
)


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
    drop_path_rate: float = 0.0
    input_format: Literal["packed", "unpacked"] = "packed"
    dropout_rate: float = 0.0


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


class Residual(Wrapper):
    """Wraps a module with a normalisation layer and residual connection.

    Args:
        component (BaseComponent): Wrapped component.
        config (ResidualConfig | None): Configuration for residual wrapper.
    """
    _base_local_attrs: ClassVar[set[str]] = (
        Wrapper._base_local_attrs
        | {
            "_config",
            "norm",
            "layer_scale",
            "drop_path_rate",
            "input_format",
            "dropout_rate",
        }
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
        self.drop_path_rate = config.drop_path_rate
        self.dropout_rate = config.dropout_rate
        self.input_format = config.input_format
        if config.norm_name == "layer":
            self.norm = nn.LayerNorm(
                self.in_features,
                elementwise_affine=False,
                eps=config.norm_eps,
            )
        elif config.norm_name == "rms":
            self.norm = nn.RMSNorm(
                self.in_features,
                elementwise_affine=False,
                eps=config.norm_eps
            )
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
        out = self.wrapped_component(self.norm(x), *args, **kwargs)

        out = maybe_apply_dropout(
            out,
            self.dropout_rate,
            self.training,
        )

        if self._config.input_format == "packed" and self.drop_path_rate > 0.0:
            assert "cu_seqlens" in kwargs, (
                "cu_seqlens must be provided for packed input format."
            )
            cu_seqlens = kwargs["cu_seqlens"]
            out = maybe_apply_stochastic_depth(
                out,
                cu_seqlens,
                self.drop_path_rate,
                self.training,
                self.input_format,
            )
        else:
            out = maybe_apply_stochastic_depth(
                out,
                None,
                self.drop_path_rate,
                self.training,
                self.input_format,
            )
        return x + out


class ResidualWithContext(Wrapper):
    """Wraps a module with a normalisation layer, residual connection, and gating.

    If context is provided, it is used for adaptive normalisation and gating.
    Gating is always initialised as zero, so the module is initially bypassed.
    """
    _base_local_attrs: ClassVar[set[str]] = (
        Wrapper._base_local_attrs
        | {
            "_context_dim",
            "_config",
            "norm",
            "scale",
            "shift",
            "layer_scale_gate",
            "drop_path_rate",
            "input_format",
            "dropout_rate",
        }
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
        self.drop_path_rate = config.drop_path_rate
        self.dropout_rate = config.dropout_rate
        self.input_format = config.input_format
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
        if hasattr(self.wrapped_component, "reset_parameters"):
            self.wrapped_component.reset_parameters()

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

        out = self.wrapped_component(tmp, *args, **kwargs) * gate
        out = maybe_apply_dropout(
            out,
            self.dropout_rate,
            self.training,
        )
        if self._config.input_format == "packed" and self.drop_path_rate > 0.0:
            assert "cu_seqlens" in kwargs, (
                "cu_seqlens must be provided for packed input format."
            )
            cu_seqlens = kwargs["cu_seqlens"]
            out = maybe_apply_stochastic_depth(
                out,
                cu_seqlens,
                self.drop_path_rate,
                self.training,
                self.input_format,
            )
        else:
            out = maybe_apply_stochastic_depth(
                out,
                None,
                self.drop_path_rate,
                self.training,
                self.input_format,
            )

        return x + out
