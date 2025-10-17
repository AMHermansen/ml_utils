import logging

logger = logging.getLogger(__name__)

from .muon import Muon

try:
    from lion_pytorch import Lion
except ImportError:
    logger.warning(
        "lion_pytorch is not installed. Please install it to use the Lion optimizer."
    )
    Lion = None


__all__ = [
    "Lion",
    "Muon",
]
