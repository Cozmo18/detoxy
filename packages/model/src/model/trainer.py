from pathlib import Path

import lightning.pytorch as pl
import torch
from jsonargparse import CLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from model.config import (
    COMET_API_KEY,
    COMET_WORKSPACE,
    Config,
    DataModuleConfig,
    ModuleConfig,
    TrainerConfig,
)
from model.datamodule import TokenizerDataModule
from model.module import ToxicClassifier
from model.utils import create_dirs, make_exp_name

# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


def train(
    model_name: str = ModuleConfig().model_name,
    learning_rate: float = ModuleConfig().learning_rate,
    max_seq_length: int = DataModuleConfig().max_seq_length,
    batch_size: int = DataModuleConfig().batch_size,
    accelerator: str = TrainerConfig().accelerator,
    devices: int | str = TrainerConfig().devices,
    strategy: str = TrainerConfig().strategy,
    precision: str | None = TrainerConfig().precision,
    max_epochs: int = TrainerConfig().max_epochs,
    log_every_n_steps: int | None = TrainerConfig().log_every_n_steps,
    deterministic: bool = TrainerConfig().deterministic,
    cache_dir: str | Path = Config().cache_dir,
    log_dir: str | Path = Config().log_dir,
    ckpt_dir: str | Path = Config().ckpt_dir,
    fast_dev_run: bool = False,
) -> None:
    create_dirs([log_dir, ckpt_dir])

    lit_datamodule = TokenizerDataModule(
        model_name=model_name,
        cache_dir=cache_dir,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )

    lit_model = ToxicClassifier(
        model_name=model_name,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
    )

    exp_name = make_exp_name(model_name, learning_rate, batch_size, max_seq_length)

    comet_logger = CometLogger(
        api_key=COMET_API_KEY,
        workspace=COMET_WORKSPACE,
        offline_directory=log_dir,
        project="toxyy",
        name=exp_name,
        mode="create",
    )
    comet_logger.log_hyperparams({"batch_size": batch_size})

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=exp_name,
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
        logger=comet_logger,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        deterministic=deterministic,
        fast_dev_run=fast_dev_run,
    )

    lit_trainer.fit(model=lit_model, datamodule=lit_datamodule)

    if not fast_dev_run:
        lit_trainer.test(ckpt_path="best", datamodule=lit_datamodule)


if __name__ == "__main__":
    CLI(train, as_positional=False)
