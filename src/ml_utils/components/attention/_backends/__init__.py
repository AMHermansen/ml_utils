"""Module providing common interfaces for attention functions across different backends.

There should generally be nothing of interest to import this module directly.
Generally all aspects that would be of interest are re-exported to attention/__init__.py
"""

from ._utils import FlashAttentionKWArgs
from .torch_flash_interface import torch_flash_attention_interface

try:
    from .flash_attention import common_flash_attention_interface
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "flash attention could not be imported, falling back to torch implementation."
    )
    common_flash_attention_interface = torch_flash_attention_interface

__all__ = [
    "FlashAttentionKWArgs",
    "common_flash_attention_interface",
    "torch_flash_attention_interface"
]
