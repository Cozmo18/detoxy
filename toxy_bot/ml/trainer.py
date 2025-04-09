import os
from time import perf_counter

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from jsonargparse import CLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG, TRAINER_CONFIG
from toxy_bot.ml.datamodule import AutoTokenizerDataModule
from toxy_bot.ml.module import SequenceClassificationModule
from toxy_bot.ml.utils import create_dirs, log_perf

load_dotenv()

# constants
model_name = MODULE_CONFIG.model_name
dataset_name = DATAMODULE_CONFIG.dataset_name

def train(
    cache_dir: str = CONFIG.cache_dir,
    log_dir: str = CONFIG.log_dir,
    ckpt_dir: str = CONFIG.ckpt_dir,
    perf_dir: str = CONFIG.perf_dir,
    accelerator: str = TRAINER_CONFIG.accelerator,
    devices: int | str = TRAINER_CONFIG.devices,
    strategy: str = TRAINER_CONFIG.strategy,
    precision: str | None = TRAINER_CONFIG.precision,
    max_epochs: int = TRAINER_CONFIG.max_epochs,
    lr: float = MODULE_CONFIG.learning_rate,
    batch_size: int = DATAMODULE_CONFIG.batch_size,
    max_length: int = DATAMODULE_CONFIG.max_length,
    deterministic: bool = TRAINER_CONFIG.deterministic,
    check_val_every_n_epoch: int | None = TRAINER_CONFIG.check_val_every_n_epoch,
    val_check_interval: int | float | None = TRAINER_CONFIG.val_check_interval,
    num_sanity_val_steps: int | None = TRAINER_CONFIG.num_sanity_val_steps,
    log_every_n_steps: int | None = TRAINER_CONFIG.log_every_n_steps,
    perf: bool = False,
    fast_dev_run: bool = False,
) -> None:
    torch.set_float32_matmul_precision(precision="medium")

    # Create required directories
    create_dirs([log_dir, ckpt_dir, perf_dir])

    lit_datamodule = AutoTokenizerDataModule(
        model_name=model_name,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        batch_size=batch_size,
        max_length=max_length,
    )

    lit_model = SequenceClassificationModule(
        model_name=model_name,
        learning_rate=lr,
    )
    comet_logger = CometLogger(
        project="toxy-bot",
        workspace="anitamaxvim",
    )

    # Configure ModelCheckpoint with Lightning's versioning
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="checkpoint_{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
    )

    # do not use EarlyStopping if getting perf benchmark
    # do not perform sanity checking if getting perf benchmark
    if perf:
        callbacks = [checkpoint_callback]
        num_sanity_val_steps = 0
    else:
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min"),
            checkpoint_callback,
        ]

    lit_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        deterministic=deterministic,
        logger=comet_logger,
        callbacks=callbacks,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=log_every_n_steps,
        num_sanity_val_steps=num_sanity_val_steps,
        fast_dev_run=fast_dev_run,
    )

    start = perf_counter()
    lit_trainer.fit(model=lit_model, datamodule=lit_datamodule)
    stop = perf_counter()

    if perf:
        # Use the version number from the logger for performance logging
        version = comet_logger.version
        log_perf(start, stop, lit_trainer, perf_dir, version)


if __name__ == "__main__":
    CLI(train, as_positional=False)
