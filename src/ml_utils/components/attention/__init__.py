from .attention_config import FlashAttentionKWArgs
from .packed_attention import PackedSelfAttention
from .torch_flash_interface import torch_flash_attention_interface

try:
    from ._flash import common_flash_attention_interface
except ModuleNotFoundError:
    common_flash_attention_interface = torch_flash_attention_interface

__all__ = [
    "FlashAttentionKWArgs",
    "PackedSelfAttention",
    "common_flash_attention_interface",
]
