# This file includes code adapted from:
#   https://github.com/mattcleigh/mltools
# Copyright (c) Matthew Leigh
# Licensed under the MIT License (See LICENSE file for details).
import logging
from collections.abc import Iterable

import torch as th

from ml_utils.torch_utils.misc import ParameterNoWeightDecay


logger = logging.getLogger(__name__)

class AdamWS(th.optim.AdamW):
    """AdamW optimizer where weight decay is only applied to matrices.

    Currently, the recommended method for transformers.
    See: https://github.com/karpathy/nanoGPT
    """

    def __init__(self, params: Iterable | dict, weight_decay: float = 1e-2, **kwargs):
        params = list(params)  # Ensures that checking wont delete elements
        if isinstance(params[0], tuple):  # Make it a list for generalisation
            params = [x for _, x in params]

        def exempt(p):
            return (p.dim() < 2) or isinstance(p, ParameterNoWeightDecay)

        nodecay_params = [p for p in params if exempt(p)]
        decay_params = [p for p in params if not exempt(p)]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            f"AdamWS: Applying weight decay {weight_decay} to {num_decay_params} "
            f"parameters, and 0.0 to {num_nodecay_params} parameters."
        )
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        super().__init__(optim_groups, **kwargs)
