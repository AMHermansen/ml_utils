import logging
from collections.abc import Mapping
from typing import Any

import jaxtyping as jt
import torch as th
from torch import nn
from typing_extensions import override

from ml_utils.utils import exists

logger = logging.getLogger(__name__)


class QKVLinear(nn.Module):
    """A linear layer to get qkv values.

    This module can switch between using a separate linear layer for each qkv,
    or a single linear layer that outputs all three concatenated.
    The reason for this somewhat unusual/ugly design, is that for flash attention
    the merged qkv projection is more performant, while the muon optimizer operates
    better with separate projections. See https://kellerjordan.github.io/posts/muon/.

    To have a cleaner attention interface, the swapping between split and merged
    projections is handled internally in this module.

    Tests will have to be done, if possible performance of split qkv is reproducible.
    """

    def __init__(self, in_features: int, bias: bool = False, split_qkv: bool = False):
        """Initialize QKVLinear.

        Args:
            in_features: Number of input features.
            bias: If True, include bias terms in the linear layers.
            split_qkv: If True, use separate linear layers for Q, K, V.
        """
        super().__init__()
        self._split_qkv = split_qkv
        self._in_features = in_features
        self._bias = bias

        if split_qkv:
            self._q_linear = nn.Linear(in_features, in_features, bias=bias)
            self._k_linear = nn.Linear(in_features, in_features, bias=bias)
            self._v_linear = nn.Linear(in_features, in_features, bias=bias)
            self._qkv_linear = None
        else:
            self._qkv_linear = nn.Linear(in_features, in_features * 3, bias=bias)
            self._q_linear = None
            self._k_linear = None
            self._v_linear = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_features={self._in_features}"
            f", out_features={self._in_features * (3 if not self._split_qkv else 1)}{'x3' if self._split_qkv else ''}"
            f", split_qkv={self._split_qkv})"
        )

    def forward(
        self, x: jt.Float[th.Tensor, "*batch in_features"]
    ) -> (
        tuple[
            jt.Float[th.Tensor, "*batch in_features"],
            jt.Float[th.Tensor, "*batch in_features"],
            jt.Float[th.Tensor, "*batch in_features"],
        ]
        | jt.Float[th.Tensor, "*batch 3*in_features"]
    ):
        if self._split_qkv:
            assert isinstance(self._q_linear, nn.Linear)
            assert isinstance(self._k_linear, nn.Linear)
            assert isinstance(self._v_linear, nn.Linear)
            return self._q_linear(x), self._k_linear(x), self._v_linear(x)
        assert isinstance(self._qkv_linear, nn.Linear)
        return self._qkv_linear(x)

    @property
    def split_qkv(self) -> bool:
        return self._split_qkv

    @th.no_grad()
    def set_split_qkv(self, split: bool) -> None:
        """Convert between split and merged QKV projections.

        Args:
            split: If True, use separate Q, K, V projections. If False, use a single
                merged QKV projection.
        """
        if self._split_qkv == split:
            return

        if split:
            # Merge -> Split: create q,k,v from contiguous chunks of qkv_proj
            if self._qkv_linear is None:
                raise RuntimeError("No merged projection available to split from.")

            w = self._qkv_linear.weight
            b = self._qkv_linear.bias
            out_total, in_features = w.shape
            if out_total % 3 != 0:
                raise ValueError(
                    "Merged projection output features not divisible by 3."
                )
            out_each = out_total // 3

            w_q, w_k, w_v = w.split(out_each, dim=0)
            b_q, b_k, b_v = (
                (None, None, None) if b is None else b.split(out_each, dim=0)
            )

            device = w.device
            dtype = w.dtype

            q_proj = nn.Linear(in_features, out_each, bias=exists(b)).to(
                device=device, dtype=dtype
            )
            k_proj = nn.Linear(in_features, out_each, bias=exists(b)).to(
                device=device, dtype=dtype
            )
            v_proj = nn.Linear(in_features, out_each, bias=exists(b)).to(
                device=device, dtype=dtype
            )

            q_proj.weight.data.copy_(w_q)
            k_proj.weight.data.copy_(w_k)
            v_proj.weight.data.copy_(w_v)
            if b is not None:
                assert b_q is not None
                assert b_k is not None
                assert b_v is not None
                q_proj.bias.data.copy_(b_q)  # type: ignore
                k_proj.bias.data.copy_(b_k)  # type: ignore
                v_proj.bias.data.copy_(b_v)  # type: ignore

            self._q_linear = q_proj
            self._k_linear = k_proj
            self._v_linear = v_proj
            self._qkv_linear = None
        else:
            # Split -> Merge: concatenate q,k,v into a single qkv linear
            if (
                self._q_linear is None
                or self._k_linear is None
                or self._v_linear is None
            ):
                raise RuntimeError("No split projections available to merge from.")

            wq, wk, wv = (
                self._q_linear.weight,
                self._k_linear.weight,
                self._v_linear.weight,
            )
            bq, bk, bv = self._q_linear.bias, self._k_linear.bias, self._v_linear.bias

            if not (wq.shape[1] == wk.shape[1] == wv.shape[1]):
                raise ValueError("In-features of q/k/v projections differ.")
            in_features = wq.shape[1]

            w_merge = th.cat([wq, wk, wv], dim=0)
            if (bq is None) != (bk is None) or (bk is None) != (bv is None):
                raise ValueError("Bias presence inconsistent among q/k/v projections.")
            b_merge = None if bq is None else th.cat([bq, bk, bv], dim=0)

            device = w_merge.device
            dtype = w_merge.dtype

            qkv_proj = nn.Linear(in_features, w_merge.shape[0], bias=exists(bq)).to(
                device=device, dtype=dtype
            )
            qkv_proj.weight.data.copy_(w_merge)
            if b_merge is not None:
                qkv_proj.bias.data.copy_(b_merge)

            self._qkv_linear = qkv_proj
            self._q_linear = None
            self._k_linear = None
            self._v_linear = None

        self._split_qkv = split

    def init_weights(self, init_std: float | None = None) -> None:
        """Initialize the weights of the linear layers.

        Args:
            init_std: Standard deviation for weight initialization.
                If None, uses default PyTorch initialization.
        """
        if init_std is None:
            init_std = self._in_features**-0.5

        def _init_linear(layer: nn.Linear) -> None:
            assert isinstance(init_std, float)
            nn.init.normal_(layer.weight, std=init_std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        if self._split_qkv:
            assert isinstance(self._q_linear, nn.Linear)
            assert isinstance(self._k_linear, nn.Linear)
            assert isinstance(self._v_linear, nn.Linear)
            _init_linear(self._q_linear)
            _init_linear(self._k_linear)
            _init_linear(self._v_linear)
        else:
            assert isinstance(self._qkv_linear, nn.Linear)
            _init_linear(self._qkv_linear)

    @override
    def load_state_dict(  # type: ignore
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        try:
            out = super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError as e:
            msg = str(e)
            if "weight" in msg or "bias" in msg:
                logger.warning(
                    "Failed to load state_dict with current split_qkv=%s. "
                    "Attempting to switch split_qkv and reload. Error: %s",
                    self._split_qkv,
                    msg,
                )
                self.set_split_qkv(not self._split_qkv)
                try:
                    out = super().load_state_dict(
                        state_dict, strict=strict, assign=assign
                    )
                except RuntimeError as e2:
                    logger.exception(
                        "Failed to load state_dict after switching split_qkv. "
                        "Please check the state_dict compatibility.",
                        stacklevel=1,
                    )
                    raise e2 from e
                else:
                    return out
        else:
            return out
