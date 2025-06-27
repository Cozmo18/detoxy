import lightning.pytorch as pl
import torch
from jsonargparse import CLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger

from model.config import (
    COMET_API_KEY,
    COMET_WORKSPACE,
    Config,
    DataModuleConfig,
    ModuleConfig,
    TrainerConfig,
)
from model.datamodule import DatasetDataModule
from model.module import SequenceClassificationModule
from model.utils import create_dirs

# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")

cache_dir = Config.cache_dir
ckpt_dir = Config.ckpt_dir

create_dirs([cache_dir, ckpt_dir])


def train(
    model_name: str = ModuleConfig.model_name,
    *,
    learning_rate: float = ModuleConfig.learning_rate,
    max_token_len: int = ModuleConfig.max_token_len,
    val_size: float = DataModuleConfig.val_size,
    batch_size: int = DataModuleConfig.batch_size,
    accelerator: str = TrainerConfig.accelerator,
    devices: int | str = TrainerConfig.devices,
    strategy: str = TrainerConfig.strategy,
    precision: str | None = TrainerConfig.precision,
    max_epochs: int = TrainerConfig.max_epochs,
    fast_dev_run: bool = False,
) -> None:
    # TODO: Think about cache and data dirs
    lit_datamodule = DatasetDataModule(
        val_size=val_size,
        batch_size=batch_size,
    )

    lit_model = SequenceClassificationModule(
        model_name=model_name,
        max_token_len=max_token_len,
        learning_rate=learning_rate,
    )

    comet_logger = CometLogger(
        api_key=COMET_API_KEY,
        workspace=COMET_WORKSPACE,
        project="nm-trainer",
    )
    comet_logger.log_hyperparams(
        {
            "model_name": model_name,
            "max_token_len": max_token_len,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val-loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    callbacks = [
        EarlyStopping(monitor="val-loss", mode="min"),
        checkpoint_callback,
    ]

    lit_trainer = pl.Trainer(
        logger=[comet_logger],
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        fast_dev_run=fast_dev_run,
        deterministic=True,
    )

    lit_trainer.fit(model=lit_model, datamodule=lit_datamodule)

    if not fast_dev_run:
        lit_trainer.test(ckpt_path="best", datamodule=lit_datamodule)


if __name__ == "__main__":
    CLI(train, as_positional=False)
