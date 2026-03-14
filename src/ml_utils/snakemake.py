"""Module containing functionality to help with Snakemake workflows."""
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback
from typing_extensions import override

if TYPE_CHECKING:
    from lightning.pytorch import Trainer, LightningModule
    import lightning.pytorch as pl


class SnakemakeConfirmFinish(Callback):
    """A callback to confirm the completion of a training run in Snakemake.

    This callback is used to signal that the training process has finished successfully.
    """

    def __init__(
        self,
        output_file: str | Path,
        msg: str | None = None,
    ):
        """Constructs the SnakemakeConfirmFinish callback.

        A callback to help snakemake confirm that a training has finished successfully.

        Args:
            output_file: The path to the output file that will be created to signal
                completion.
            msg: A message to write into the output file. If None, a default message is
                used.
        """
        super().__init__()
        self._output_file = Path(output_file)
        self._msg = (
            msg
            if msg is not None
            else "Run completed successfully.\nOnly needed for snakemake.\n"
        )

    def _write_confirmation(self):
        if not self._output_file.parent.exists():
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
        with self._output_file.open("w") as f:
            f.write(self._msg)
    
    @override
    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Writes the confirmation file upon training completion.

        Args:
            trainer: The Trainer instance.
            pl_module: The LightningModule instance.
            stage: The stage of training ('fit', 'validate', 'test', or 'predict').
        """
        self._write_confirmation()

def process_additional_params(prefix: str, **kwargs) -> list[str]:
    """Processes additional parameters for command generation.

    Args:
        prefix: The prefix to use for the parameters.
        **kwargs: Additional keyword arguments to process.

    Returns:
        A list of strings representing the additional parameters in command format.
    """
    params = []
    for key, value in kwargs.items():
        if isinstance(value, str):
            value_str = f'"{value}"'
        else:
            value_str = str(value)
        params.append(f"--{prefix}.{key}={value_str}")
    return params


def add_snakemake_confirm_callback_command(output_file: str | Path) -> str:
    return " ".join(
        [
            f"--trainer.callbacks+={SnakemakeConfirmFinish.__module__}.{SnakemakeConfirmFinish.__name__}",
            f"--trainer.callbacks.output_file={output_file}",
        ]
    )


def add_checkpoint_command(
    dir_path: Path,
    monitor: str,
    mode: str = "min",
    filename: str = "{epoch:02d}",
    save_top_k: int = 3,
    **kwargs,
) -> str:
    """Generates a command string to add a ModelCheckpoint callback to a Snakemake workflow.

    Args:
        dir_path: The directory path where checkpoints will be saved.
        monitor: The metric name to monitor for saving checkpoints.
        mode: The mode for monitoring ('min' or 'max').
        filename: The filename template for saved checkpoints.
        save_top_k: The number of top checkpoints to save.
        **kwargs: Additional keyword arguments to pass to the ModelCheckpoint callback.

    Returns:
        A command string to be used in a Snakemake workflow for adding the
        ModelCheckpoint callback

    See Also:
        lightning.pytorch.callbacks.ModelCheckpoint
    """
    return " ".join(
        [
            "--trainer.callbacks+=lightning.pytorch.callbacks.ModelCheckpoint",
            f"--trainer.callbacks.filename='{filename}'",
            f"--trainer.callbacks.monitor={monitor}",
            f"--trainer.callbacks.mode={mode}",
            f"--trainer.callbacks.dirpath={dir_path!s}",
            f"--trainer.callbacks.save_top_k={save_top_k}",
        ] + process_additional_params("trainer.callbacks", **kwargs)
    )

# Convenience partial functions for common checkpoint configurations
add_best_checkpoint_command = partial(add_checkpoint_command, save_top_k=1, filename="best")


def add_early_stopping_command(
    monitor: str,
    patience: int,
    mode: str = "min",
    **kwargs,
) -> str:
    """Generates a command string to add an EarlyStopping callback to a Snakemake workflow.

    Args:
        monitor: The metric name to monitor for early stopping.
        patience: The number of epochs with no improvement after which training will be stopped.
        mode: The mode for monitoring ('min' or 'max').
        **kwargs: Additional keyword arguments to pass to the EarlyStopping callback.

    Returns:
        A command string to be used in a Snakemake workflow for adding the
        EarlyStopping callback.

    See Also:
        lightning.pytorch.callbacks.EarlyStopping
    """
    return " ".join(
        [
            "--trainer.callbacks+=lightning.pytorch.callbacks.EarlyStopping",
            f"--trainer.callbacks.monitor={monitor}",
            f"--trainer.callbacks.patience={patience}",
            f"--trainer.callbacks.mode={mode}",
        ] + process_additional_params("trainer.callbacks", **kwargs)
    )


def add_learning_rate_monitor_command(
    logging_interval: str = "step",
    **kwargs,
) -> str:
    """Generates a command string to add a LearningRateMonitor callback to a Snakemake workflow.

    Args:
        logging_interval: The interval at which to log the learning rate ('step' or 'epoch').
        **kwargs: Additional keyword arguments to pass to the LearningRateMonitor callback.

    Returns:
        A command string to be used in a Snakemake workflow for adding the
        LearningRateMonitor callback.

    See Also:
        lightning.pytorch.callbacks.LearningRateMonitor
    """
    return " ".join(
        [
            "--trainer.callbacks+=lightning.pytorch.callbacks.LearningRateMonitor",
            f"--trainer.callbacks.logging_interval={logging_interval}",
        ] + process_additional_params("trainer.callbacks", **kwargs)
    )


def add_wandb_logger_command(
    project: str,
    name: str,
    save_dir: Path,
    tags: list[str] | None = None,
    **kwargs,
) -> str:
    """Generates a command string to add a WandbLogger to a Snakemake workflow.

    Args:
        project: The name of the Weights & Biases project.
        name: The name of the experiment/run.
        save_dir: The directory where logs will be saved.
        tags: A list of tags to associate with the run.
        **kwargs: Additional keyword arguments to pass to the WandbLogger.

    Returns:
        A command string to be used in a Snakemake workflow for adding the
        WandbLogger.
    """
    tags_command_prefix = "--trainer.logger.init_args.tags"
    tags_command = " ".join([f"{tags_command_prefix}+='{tag}'" for tag in tags] if tags else "")
    return " ".join(
        [
            f"--trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger",
            f"--trainer.logger.init_args.project={project}",
            f"--trainer.logger.init_args.name={name}",
            f"--trainer.logger.init_args.save_dir={save_dir}",
            f"{tags_command}",
        ] + process_additional_params("trainer.logger.init_args", **kwargs)
    )


def add_csv_logger_command(save_dir: Path) -> str:
    """Generates a command string to add a CSVLogger to a Snakemake workflow.

    Args:
        save_dir: The directory where logs will be saved.

    Returns:
        A command string to be used in a Snakemake workflow for adding the
        CSVLogger.
    """
    return " ".join(
        [
            f"--trainer.logger.class_path=lightning.pytorch.loggers.CSVLogger",
            f"--trainer.logger.init_args.save_dir={save_dir}",
            f"--trainer.logger.init_args.name=csv_logs",
        ]
    )
