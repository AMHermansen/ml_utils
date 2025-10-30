import logging

logger = logging.getLogger(__name__)

from .adamws import AdamWS
from .muon import Muon, suitable_for_muon
from .schedulers import LinearWarmupCosineDecay

try:
    from lion_pytorch import Lion
except ImportError:
    logger.warning(
        "lion_pytorch is not installed. Please install it to use the Lion optimizer."
    )
    Lion: None = None


__all__ = [
    "AdamWS",
    "LinearWarmupCosineDecay",
    "Lion",
    "Muon",
    "suitable_for_muon",
]
