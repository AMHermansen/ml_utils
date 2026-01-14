from dataclasses import dataclass, field
from typing import Literal

from ml_utils.components.attention.attention_config import BiasAttentionConfig
from ml_utils.components.mlp import MLPBlockConfig


@dataclass
class TriangleMultiplicationConfig:
    """Configuration for triangle multiplication operation.

    Attributes:
        norm_type: Type of normalization to use. Options are "layer", "rms", or None.
        use_bias: Whether to include a bias term in the multiplication.
    """

    norm_type: Literal["layer", "rms"] | None = None
    use_bias: bool = False


@dataclass
class TriangleAttentionConfig:
    """Configuration for triangle attention operation.

    Attributes:
        norm_type: Type of normalization to use. Options are "layer", "rms", or None.
        use_bias: Whether to include a bias term in the attention mechanism.
        num_heads: Number of attention heads to use.
        use_flex_attention: Whether to use flexible attention mechanism.
    """

    norm_type: Literal["layer", "rms"] | None = None
    use_bias: bool = False
    num_heads: int = 4
    use_flex_attention: bool = False


@dataclass
class PairFormerBlockConfig:
    """Configuration for a single PairFormer block."""

    triangle_multiplication_config: TriangleMultiplicationConfig = field(
        default_factory=TriangleMultiplicationConfig
    )
    triangle_attention_config: TriangleAttentionConfig = field(
        default_factory=TriangleAttentionConfig
    )
    single_attention_config: BiasAttentionConfig = field(
        default_factory=BiasAttentionConfig
    )
    pair_mlp_config: MLPBlockConfig = field(default_factory=MLPBlockConfig)
    single_mlp_config: MLPBlockConfig = field(default_factory=MLPBlockConfig)

    dropout_probability: float = 0.25
    compile_modules: bool = False


@dataclass
class PairFormerConfig:
    """Configuration for the PairFormer model.

    Attributes:
        num_blocks: Number of PairFormer blocks to stack.
        block_config: Configuration for each PairFormer block.
    """

    num_blocks: int = 4
    block_config: PairFormerBlockConfig = field(default_factory=PairFormerBlockConfig)
    