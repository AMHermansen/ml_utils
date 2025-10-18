from .attention_config import (
    CrossAttentionConfig,
    FlashAttentionKWArgs,
    SelfAttentionConfig,
)
from .packed_cross_attention import PackedCrossAttention
from .packed_self_attention import PackedSelfAttention
from .torch_flash_interface import torch_flash_attention_interface

try:
    from ._flash import common_flash_attention_interface
except ModuleNotFoundError:
    common_flash_attention_interface = torch_flash_attention_interface

__all__ = [
    "CrossAttentionConfig",
    "FlashAttentionKWArgs",
    "PackedCrossAttention",
    "PackedSelfAttention",
    "SelfAttentionConfig",
    "common_flash_attention_interface",
]
