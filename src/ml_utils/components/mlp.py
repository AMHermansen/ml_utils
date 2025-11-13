from dataclasses import dataclass, field
from typing import Literal, overload

import torch as th
from torch import nn
from typing_extensions import override

from ml_utils.torch_utils import append_dimensions
from ml_utils.utils import exists

from .base import BaseComponent


@dataclass
class MLPBlockConfig:
    """General configuration for MLP block.

    Attributes:
        activation: Activation function to use (string name or nn.Module).
        norm: Normalization layer to use (string name, nn.Module, or None).
        dropout: Dropout probability (0.0 means no dropout).
        do_residual: Whether to include a residual connection.
        no_bias: Whether to disable bias in the linear layer.
    """
    activation: str | nn.Module = "SiLU"
    norm: str | nn.Module | None = None
    dropout: float = 0.0
    do_residual: bool = False
    no_bias: bool = False


@dataclass
class MLPContextConfig:
    """Configuration for incorporating context features into MLP.

    Attributes:
        context_features: Number of context features to incorporate.
        apply_on_input: Whether to apply context on input layer.
        apply_on_hidden: Whether to apply context on hidden layers.
        apply_on_output: Whether to apply context on output layer.
    """
    context_features: int = 0
    apply_on_input: bool = True
    apply_on_hidden: bool = False
    apply_on_output: bool = False

    def __post_init__(self):
        if self.context_features and not (
            self.apply_on_input
            or self.apply_on_hidden
            or self.apply_on_output
        ):
            raise ValueError(
                "At least one of apply_on_input, apply_on_hidden, or apply_on_output"
                " must be True if context_features is specified."
            )


@dataclass
class MLPConfig:
    """General configuration for MLP.

    Attributes:
        block_config: Configuration for MLP blocks.
        context_config: Configuration for context features.
        hidden_features: Number of hidden features in the MLP. If None, defaults to
            out_features.
    """
    block_config: MLPBlockConfig = field(default_factory=MLPBlockConfig)
    context_config: MLPContextConfig = field(default_factory=MLPContextConfig)
    num_layers: int = 2
    hidden_features: int | None = None


class MLPBlock(BaseComponent):
    """Linear layer with optional context, activation, normalization, dropout.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        config: Configuration for the MLP block.
        context_features: Number of context features to concatenate to input.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: MLPBlockConfig,
        context_features: int = 0
    ):
        """Linear layer with optional context, activation, normalization, dropout.

        This block implements a linear transformation with optional context
        concatenation, followed by an activation function, normalization layer,
        and dropout. It can also include a residual connection if specified.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            config: Configuration for the MLP block.
            context_features: Number of context features to concatenate to input.
        """
        super().__init__()
        if config.do_residual and in_features != out_features:
            raise ValueError("in_features must be equal to out_features for residual "
                             "connection")

        self._in_features = in_features
        self._out_features = out_features
        self._context_features = context_features
        self._config = config
        self._do_residual = config.do_residual

        self.linear = nn.Linear(
            in_features + context_features,
            out_features,
            bias=not config.no_bias,
        )
        self.activation = self._get_activation(config.activation)
        self.norm = self._get_norm(config.norm, out_features)
        self.dropout = (
            nn.Dropout(config.dropout)
            if config.dropout > 0.0
            else nn.Identity()
        )

    def forward(
        self,
        x: th.Tensor,
        *,
        context: th.Tensor | None = None,
    ):
        """Forward pass through the MLP block.

        Args:
            x: Input tensor of shape (..., in_features).
            context: Optional context tensor of shape (..., context_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Handling potential context mismatch
        if self.expects_context and not exists(context):
            raise ValueError("Context features expected but context is None")
        if not self.expects_context and exists(context):
            raise ValueError("Context features not expected but context is provided")

        if exists(context):
            # If batch-like dimensions do not match, refuse the temptation to guess.
            if context.shape[0] != x.shape[0]:
                raise ValueError("Cannot resolve dimensional mismatch between x and "
                                 "context on first dimension."
                                 f" Got {x.shape=} and {context.shape=}")
            context = append_dimensions(context, x.ndim, 1).expand(
                -1,
                *x.shape[1:-1],
                -1
            )
            x_maybe_cat = th.cat([x, context], dim=-1)
        else:
            x_maybe_cat = x
        out = self.linear(x_maybe_cat)
        out = self.activation(out)
        out = self.norm(out)
        out = self.dropout(out)
        if self._do_residual:
            out = out + x
        return out

    # helper methods
    def _get_activation(self, activation: str | nn.Module) -> nn.Module:
        if isinstance(activation, nn.Module):
            return activation
        try:
            return getattr(nn, activation)()
        except AttributeError as err:
            raise ValueError(
                f"Unsupported activation: {activation}, it must be either a string name"
                " of a torch.nn activation or an instance of nn.Module."
            ) from err

    def _get_norm(self, norm: str | nn.Module | None, features: int) -> nn.Module:
        if norm is None:
            return nn.Identity()
        if isinstance(norm, nn.Module):
            return norm
        try:
            norm_class = getattr(nn, norm)
            return norm_class(features)
        except AttributeError as err:
            raise ValueError(
                f"Unsupported norm: {norm}, it must be either a string name of a"
                " torch.nn normalization layer, an instance of nn.Module, or None."
            ) from err

    # properties
    @property
    def expects_context(self) -> bool:
        """Whether this block expects context features."""
        return self._context_features > 0

    # abstract properties implementation
    @property
    @override
    def out_features(self) -> int:
        return self._out_features

    @property
    @override
    def in_features(self) -> int:
        return self._in_features

    @override
    def __repr__(self):
        string = str(self._in_features)
        if self.expects_context:
            string += f"(+{self._context_features})"
        string += f" -> {self._out_features}"
        activation_str = (
            ""
            if isinstance(self.activation, nn.Identity)
            else f"-> {str(self.activation).split('(', 1)[0]}"
        )
        norm_str = (
            ""
            if isinstance(self.norm, nn.Identity)
            else f"-> {str(self.norm).split('(', 1)[0]}"
        )
        dropout_str = (
            ""
            if isinstance(self.dropout, nn.Identity)
            else f"-> Dropout(p={self._config.dropout})"
        )
        string = f"MLPBlock({string} {activation_str} {norm_str} {dropout_str}"
        if self._do_residual:
            string += " | Residual"
        string += ")"
        return string


class MLP(BaseComponent):
    """Multi-layer perceptron (MLP) with optional context features.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        config: Configuration for the MLP.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: MLPConfig | None = None,
    ):
        """Multi-layer perceptron (MLP) with optional context features.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            config: Configuration for the MLP.
        """
        super().__init__()
        config = config if exists(config) else MLPConfig()
        hidden_features = (
            config.hidden_features
            if exists(config.hidden_features)
            else out_features
        )
        self._in_features = in_features
        self._out_features = out_features
        self._hidden_features = hidden_features
        self._config = config
        self._num_layers = config.num_layers
        self._context_config = (
            config.context_config
        )
        self._block_config = config.block_config
        self._check_valid_context()
        self._setup_layers()

    def forward(
        self,
        x: th.Tensor,
        *,
        context: th.Tensor | None = None
    ):
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (..., in_features).
            context: Optional context tensor of shape (..., context_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        out = self._in_block(
            x,
            context=self._get_maybe_context(
                context,
                self._context_config.apply_on_input,
            )
        )
        for block in self._hidden_blocks:
            out = block(
                out,
                context=self._get_maybe_context(
                    context,
                    self._context_config.apply_on_hidden,
                )
            )
        return self._out_block(
            out,
            context=self._get_maybe_context(
                context,
                self._context_config.apply_on_output,
            )
        )

    def _check_valid_context(self):
        # Early return if no context is used
        if self._context_config.context_features == 0 and not (
            self._context_config.apply_on_input
            or self._context_config.apply_on_hidden
            or self._context_config.apply_on_output
        ):
            return
        using_context = self._context_config.context_features > 0

        if using_context and self._num_layers == 1 and not self._context_config.apply_on_input:
            raise ValueError(
                "With only one layer, context can only be applied on input."
            )
        if using_context and self._num_layers == 2 and not (
            self._context_config.apply_on_input
            or self._context_config.apply_on_output
        ):
            raise ValueError(
                "With two layers, context can only be applied on input or output."
            )

    def _setup_layers(self):
        self._in_block = MLPBlock(
            in_features=self._in_features,
            out_features=self._hidden_features,
            context_features=(
                self._context_config.context_features
                if self._context_config.apply_on_input
                else 0
            ),
            config=self._block_config,
        )
        self._hidden_blocks = nn.ModuleList(
            [
                MLPBlock(
                    in_features=self._hidden_features,
                    out_features=self._hidden_features,
                    context_features=(
                        self._context_config.context_features
                        if self._context_config.apply_on_hidden
                        else 0
                    ),
                    config=self._block_config,
                )
                for _ in range(self._num_layers - 2)
            ]
        )
        self._out_block = MLPBlock(
            in_features=self._hidden_features,
            out_features=self._out_features,
            context_features=(
                self._context_config.context_features
                if self._context_config.apply_on_output
                else 0
            ),
            config=self._block_config,
        )

    @overload
    def _get_maybe_context(self, context: None, should_apply: bool) -> None: ...

    @overload
    def _get_maybe_context(
        self,
        context: th.Tensor,
        should_apply: Literal[True]
    ) -> th.Tensor:
        ...

    @overload
    def _get_maybe_context(
        self,
        context: th.Tensor,
        should_apply: Literal[False]
    ) -> None:
        ...

    def _get_maybe_context(
        self,
        context: th.Tensor | None,
        should_apply: bool
    ) -> th.Tensor | None:
        """Helper to get correctly select context or None."""
        if should_apply:
            return context
        return None

    @property
    @override
    def out_features(self) -> int:
        return self._out_features

    @property
    @override
    def in_features(self) -> int:
        return self._in_features
