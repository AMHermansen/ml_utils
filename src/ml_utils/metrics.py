import jaxtyping as jt
import torch as th
from torchmetrics import Metric


class RejectionAtFixedEfficiency(Metric):
    def __init__(
        self,
        signal_indices: list[int] | int,
        background_indices: list[int] | int,
        target_efficiency: float = 0.9,
        from_logits: bool = True,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.signal_indices = (
            signal_indices if isinstance(signal_indices, list) else [signal_indices]
        )
        self.background_indices = (
            background_indices
            if isinstance(background_indices, list)
            else [background_indices]
        )
        self.target_efficiency = target_efficiency
        self.from_logits = from_logits

        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    @override
    def update(
        self,
        preds: jt.Float[th.Tensor, " batch_size num_classes"],
        targets: jt.Long[th.Tensor, " batch_size"],
    ) -> None:
        if self.from_logits:
            preds = th.softmax(preds, dim=-1)
        self.predictions.append(preds)
        self.targets.append(targets)

    @override
    def compute(self) -> float:
        preds = th.cat(self.predictions, dim=0)
        targets = th.cat(self.targets, dim=0)

        signal_mask = th.isin(
            targets, th.tensor(self.signal_indices, device=targets.device)
        )
        background_mask = th.isin(
            targets, th.tensor(self.background_indices, device=targets.device)
        )

        if signal_mask.sum() == 0 or background_mask.sum() == 0:
            return float("nan")

        sorted_scores, _ = th.sort(preds[signal_mask], dim=0, descending=True)
        threshold_index = int(self.target_efficiency * signal_mask.sum())
        threshold_index = min(threshold_index, signal_mask.sum() - 1)
        threshold = sorted_scores[threshold_index]

        background_efficiency = (
            (preds[background_mask] >= threshold).float().mean().item()
        )
        if background_efficiency == 0:
            return float("inf")
        return 1.0 / background_efficiency
