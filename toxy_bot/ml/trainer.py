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
from toxy_bot.ml.utils import create_dirs, create_experiment_name, log_perf

load_dotenv()

# Config instances
# constants
model_name = MODULE_CONFIG.model_name
dataset_name = DATAMODULE_CONFIG.dataset_name

# paths
cache_dir = CONFIG.cache_dir
log_dir = CONFIG.log_dir
ckpt_dir = CONFIG.ckpt_dir
perf_dir = CONFIG.perf_dir


def train(
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

    # Create unique run/experiment name
    experiment_name = create_experiment_name(
        model_name=model_name,
        learning_rate=lr,
        batch_size=batch_size,
        max_length=max_length,
    )

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
        api_key=os.getenv("COMET_API_KEY"),
        project="toxy-bot",
        workspace="anitamaxvim",
        name=experiment_name,
    )

    # do not use EarlyStopping if getting perf benchmark
    # do not perform sanity checking if getting perf benchmark
    if perf:
        callbacks = [ModelCheckpoint(dirpath=ckpt_dir, filename=experiment_name)]
        num_sanity_val_steps = 0
    else:
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(dirpath=ckpt_dir, filename=experiment_name),
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
        log_perf(start, stop, lit_trainer, perf_dir, experiment_name)


if __name__ == "__main__":
    CLI(train, as_positional=False)
