from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback


class WatchModel(Callback):
    # Define a model monitoring callback class used to register the model to the logger
    def __init__(self, log_freq: int = 100):
        super().__init__()
        # Set the logging frequency
        self.log_freq = log_freq

    # Start monitoring the model after the sanity check ends
    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)
        # Let the logger watch all model information, such as parameters and gradients
        trainer.logger.watch(pl_module, log="all", log_freq=self.log_freq)