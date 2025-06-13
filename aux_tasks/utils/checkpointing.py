from dataclasses import dataclass
import os
import pathlib

from flax.training import orbax_utils
import orbax.checkpoint


@dataclass
class CheckpointHandler:
    checkpoint_dir: str

    def __post_init__(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=2, create=True
        )
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            os.path.abspath(self.checkpoint_dir), self.orbax_checkpointer, self.options
        )

    def checkpoint_params(self, model, step: int):
        self.checkpoint_manager.save(step, model)

    def restore_params(self, models, path: str):
        step = self.checkpoint_manager.latest_step()  # step = 4
        return self.checkpoint_manager.restore(step, items=models)
