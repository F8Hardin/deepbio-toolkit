from lightning.pytorch.callbacks import Callback
from typing import Any, Dict

class ConfigCheckpoint(Callback):
    """Callback to save config in checkpoint."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['config'] = self.config

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'config' in checkpoint:
            self.config = checkpoint['config']


class WandbResumeCallback(Callback):
    """
    Persists the active W&B run ID into the ConfigCheckpoint config on training start.

    On the first checkpoint save after this runs, the raw config dict will contain
    the run's id and resume="must", so that any subsequent resume of the Lightning
    checkpoint automatically continues the same W&B run rather than starting a new one.
    """

    def __init__(self, config_checkpoint: ConfigCheckpoint):
        super().__init__()
        self.config_checkpoint = config_checkpoint

    def on_train_start(self, trainer, pl_module):
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError:
            return

        wandb_logger = next(
            (l for l in trainer.loggers if isinstance(l, WandbLogger)), None
        )
        if wandb_logger is None:
            return

        run_id = wandb_logger.experiment.id
        if not run_id:
            return

        # Patch the raw config dict so future checkpoints carry the run ID.
        # The logger section is at config['train']['logger'] as a class_path dict.
        train_cfg = self.config_checkpoint.config.get('train', {})
        logger_raw = train_cfg.get('logger', {})
        if isinstance(logger_raw, dict) and 'class_path' in logger_raw:
            logger_raw.setdefault('config', {}).update({
                'id': run_id,
                'resume': 'must',
            })
