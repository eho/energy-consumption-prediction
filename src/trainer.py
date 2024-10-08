# main.py
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from .dataset import EnergyConsumptionDataModule
from .model import LSTMModel


class EnergyComsumptionModelLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,  # Save only the best model
            verbose=True,  # Print out when the checkpoint is saved
            mode="min",
            monitor="val_loss",
            dirpath=None,  # Save in the default 'lightning_logs/version_x' directory
            filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        )
        # Pass the callback when initializing the Trainer
        super().__init__(
            *args,
            trainer_defaults={
                "callbacks": [checkpoint_callback]  # Add the ModelCheckpoint callback
            },
            **kwargs,
        )

    def after_fit(self):
        # Print the path to the best checkpoint after fitting
        checkpoint_callback = self.trainer.checkpoint_callback
        if checkpoint_callback is not None and checkpoint_callback.best_model_path:
            print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")
        else:
            print("No checkpoint callback or no checkpoint saved.")


def cli_main():
    cli = EnergyComsumptionModelLightningCLI(LSTMModel, EnergyConsumptionDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
