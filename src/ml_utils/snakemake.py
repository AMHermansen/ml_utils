"""Module containing functionality to help with Snakemake workflows."""
import inspect
from pathlib import Path

from lightning.pytorch.callbacks import Callback
from typing_extensions import override


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
            else "Training completed successfully.\nOnly needed for snakemake.\n"
        )

    @override
    def on_train_end(self, trainer, pl_module):
        """Called when the training ends.

        It creates an 'output' file for snake to verify a successful run.
        """
        if not self._output_file.parent.exists():
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
        with self._output_file.open("w") as f:
            f.write(self._msg)


def add_snakemake_config_callback_command(output_file: str | Path, msg: str | None = None) -> str:
    return f"--trainer.callbacks+={SnakemakeConfirmFinish.__module__}.{SnakemakeConfirmFinish.__name__}"
