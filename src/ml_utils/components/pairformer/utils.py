from dataclasses import dataclass, field
from typing import Literal

from ml_utils.components.attention.attention_config import BiasAttentionConfig
from ml_utils.components.mlp import MLPBlockConfig, MLPConfig


@dataclass
class TriangleMultiplicationConfig:
    """Configuration for triangle multiplication operation.

    Attributes:
        norm_type: Type of normalization to use. Options are "layer", "rms", or None.
            Using no normalization is not recommended, and can lead to training
            instability.
        use_bias: Whether to include a bias term in the multiplication.
    """

    norm_type: Literal["layer", "rms"] | None = "rms"
    use_bias: bool = False


@dataclass
class TriangleAttentionConfig:
    """Configuration for triangle attention operation.

    Attributes:
        norm_type: Type of normalization to use. Options are "layer", "rms", or None.
            Using no normalization is not recommended, and can lead to training
            instability.
        use_bias: Whether to include a bias term in the attention mechanism.
        num_heads: Number of attention heads to use.
        use_flex_attention: Whether to use flexible attention mechanism.
    """

    norm_type: Literal["layer", "rms"] | None = "rms"
    use_bias: bool = False
    num_heads: int = 4
    use_flex_attention: bool = False


@dataclass
class PairFormerBlockConfig:
    """Configuration for a single PairFormer block.

    Args:
        triangle_multiplication_config: Configuration for the triangle multiplication operation.
        triangle_attention_config: Configuration for the triangle attention operation.
        single_attention_config: Configuration for the single attention operation.
        pair_mlp_config: Configuration for the MLP applied to pair features.
        single_mlp_config: Configuration for the MLP applied to single features.
        dropout_probability: Dropout probability to apply after each operation.
        use_pre_mlp_norm: Whether to apply RMS Norm before the MLPs.
        compile_modules: This is the preferred way to compile the pairformer block.
            The reason directly compiling the block is not recommended is due to a bug in
            the bias attention module which causes run-time errors when compiled.
    """

    triangle_multiplication_config: TriangleMultiplicationConfig = field(
        default_factory=TriangleMultiplicationConfig
    )
    triangle_attention_config: TriangleAttentionConfig = field(
        default_factory=TriangleAttentionConfig
    )
    single_attention_config: BiasAttentionConfig = field(
        default_factory=BiasAttentionConfig
    )
    pair_mlp_config: MLPConfig = field(default_factory=MLPConfig)
    single_mlp_config: MLPConfig = field(default_factory=MLPConfig)

    dropout_probability: float = 0.25
    use_pre_mlp_norm: bool = False
    compile_modules: bool = False


@dataclass
class PairFormerConfig:
    """Configuration for the PairFormer model.

    Attributes:
        num_layers: Number of PairFormer blocks to stack.
        block_config: Configuration for each PairFormer block.
    """

    num_layers: int = 4
    block_config: PairFormerBlockConfig = field(default_factory=PairFormerBlockConfig)
