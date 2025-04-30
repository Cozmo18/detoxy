import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from jsonargparse import CLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG, TRAINER_CONFIG
from toxy_bot.ml.datamodule import AutoTokenizerDataModule
from toxy_bot.ml.module import SequenceClassificationModule
from toxy_bot.ml.utils import create_dirs, make_exp_name


# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


def train(
    model_name: str = MODULE_CONFIG.model_name,
    learning_rate: float = MODULE_CONFIG.learning_rate,
    max_seq_length: int = DATAMODULE_CONFIG.max_seq_length,
    batch_size: int = DATAMODULE_CONFIG.batch_size,
    accelerator: str = TRAINER_CONFIG.accelerator,
    devices: int | str = TRAINER_CONFIG.devices,
    strategy: str = TRAINER_CONFIG.strategy,
    precision: str | None = TRAINER_CONFIG.precision,
    max_epochs: int = TRAINER_CONFIG.max_epochs,
    log_every_n_steps: int | None = TRAINER_CONFIG.log_every_n_steps,
    deterministic: bool = TRAINER_CONFIG.deterministic,
    cache_dir: str | Path = CONFIG.cache_dir,
    log_dir: str | Path = CONFIG.log_dir,
    ckpt_dir: str | Path = CONFIG.ckpt_dir,
    fast_dev_run: bool = False,
) -> None:
    create_dirs([log_dir, ckpt_dir])

    lit_datamodule = AutoTokenizerDataModule(
        model_name=model_name,
        cache_dir=cache_dir,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )

    lit_model = SequenceClassificationModule(
        model_name=model_name,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
    )

    exp_name = make_exp_name(model_name, learning_rate, batch_size, max_seq_length)

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        workspace=os.environ.get("COMET_WORKSPACE"),
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
