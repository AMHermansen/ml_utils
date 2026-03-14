"""Modules and utilities for PyTorch Lightning integration."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch as th
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override

from ml_utils.torch_utils.optim import Muon, suitable_for_muon
from ml_utils.utils import exists

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import (
        OptimizerLRScheduler,
        OptimizerLRSchedulerConfig,
    )


class WandBSaveConfigCallback(SaveConfigCallback):
    @override
    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if isinstance(trainer.logger, WandbLogger):
            logger = cast("WandbLogger", trainer.logger)
            assert trainer.log_dir is not None, "Trainer log_dir is None."
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
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    scheduler_class: type[th.optim.lr_scheduler.LRScheduler] | None = None
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    scheduler_config: dict[str, Any] = field(default_factory=dict)


def configure_optimizer_standard(
    model: LightningModule, lightning_config: LightningConfig
) -> "OptimizerLRSchedulerConfig":
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
        config.update({  # type: ignore
            "lr_scheduler": {  # type: ignore
                "scheduler": scheduler,
                **lightning_config.scheduler_config,
            }
        })
    return cast("OptimizerLRSchedulerConfig", config)


def configure_muon_optimizer(
    muon_parameter_candidates: Iterable[th.nn.Parameter],
    remaining_parameters: Iterable[th.nn.Parameter],
    lightning_config: LightningConfig,
):
    """Configure optimizer and scheduler for Muon optimizer.

    This function sets up the optimizer and learning rate scheduler
    specifically for Muon models, using the provided Lightning configuration.
    """
    muon_params = filter(
        lambda p: p.requires_grad and suitable_for_muon(p), muon_parameter_candidates
    )
    adam_params1 = filter(
        lambda p: p.requires_grad and not suitable_for_muon(p), muon_params
    )
    adam_params2 = filter(lambda p: p.requires_grad, remaining_parameters)

    muon_group = {
        "params": muon_params,
        "lr": lightning_config.optimizer_kwargs.get(
            "muon_lr",
            0.001,
        ),
        "momentum": lightning_config.optimizer_kwargs.get(
            "muon_momentum",
            0.95,
        ),
        "weight_decay": lightning_config.optimizer_kwargs.get(
            "muon_weight_decay",
            0.01,
        ),
        "use_muon": True,
    }

    adam_group = {
        "params": chain(adam_params1, adam_params2),
        "lr": lightning_config.optimizer_kwargs.get(
            "adam_lr",
            3e-4,
        ),
        "betas": lightning_config.optimizer_kwargs.get(
            "adam_betas",
            (0.9, 0.95),
        ),
        "eps": lightning_config.optimizer_kwargs.get(
            "adam_eps",
            1e-10,
        ),
        "weight_decay": lightning_config.optimizer_kwargs.get(
            "adam_weight_decay", 0.01
        ),
        "use_muon": False,
    }
    optimizer = Muon([muon_group, adam_group])
    config = {
        "optimizer": optimizer,
    }
    if lightning_config.scheduler_class is not None:
        scheduler = lightning_config.scheduler_class(
            optimizer, **lightning_config.scheduler_kwargs
        )
        config.update({  # type: ignore
            "lr_scheduler": {  # type: ignore
                "scheduler": scheduler,
                **lightning_config.scheduler_config,
            }
        })
