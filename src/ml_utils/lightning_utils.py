"""Modules and utilities for PyTorch Lightning integration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch as th
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from typing_extensions import override


class WandBSaveConfigCallback(SaveConfigCallback):
    @override
    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if isinstance(trainer.logger, WandbLogger):
            logger = cast("WandbLogger", trainer.logger)
            experiment = logger.experiment
            experiment.log_artifact(
                artifact_or_path=Path(trainer.log_dir, self.config_filename),
                name="cli_config",
            )


@dataclass
class LightningConfig:
    """Configuration for PyTorch Lightning training.

    Attributes:
        optimizer_class: The optimizer class to use (default: AdamW).
        optimizer_kwargs: Keyword arguments for the optimizer.
        scheduler_class: The learning rate scheduler class.
        scheduler_kwargs: Keyword arguments for the scheduler.
        scheduler_config: Configuration for the scheduler.
    """

    optimizer_class: type[th.optim.Optimizer] = field(default=th.optim.AdamW)
    optimizer_kwargs: dict[str, Any] = None
    scheduler_class: type[th.optim.lr_scheduler.LRScheduler] = None
    scheduler_kwargs: dict[str, Any] = None
    scheduler_config: dict[str, Any] = None


def configure_optimizer_standard(
    model: LightningModule, lightning_config: LightningConfig
) -> OptimizerLRScheduler:
    optimizer = lightning_config.optimizer_class(
        filter(lambda p: p.requires_grad, model.parameters()),
        **lightning_config.optimizer_kwargs,
    )
    config = {
        "optimizer": optimizer,
    }
    if lightning_config.scheduler_class is not None:
        scheduler = lightning_config.scheduler_class(
            optimizer, **lightning_config.scheduler_kwargs
        )
        config.update(
            {
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **lightning_config.scheduler_config,
                }
            }
        )
    return config
