import os
from time import perf_counter

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from jsonargparse import CLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

# from lightning.pytorch.loggers import CSVLogger
from toxy_bot.ml.config import Config, DataModuleConfig, ModuleConfig, TrainerConfig
from toxy_bot.ml.datamodule import AutoTokenizerDataModule
from toxy_bot.ml.module import SequenceClassificationModule
from toxy_bot.ml.utils import create_dirs, create_experiment_name, log_perf

load_dotenv()

# Config instances
config = Config()
datamodule_config = DataModuleConfig()
module_config = ModuleConfig()
trainer_config = TrainerConfig()

# constants
model_name = module_config.model_name
dataset_name = datamodule_config.dataset_name

# paths
cache_dir = Config.cache_dir
log_dir = Config.log_dir
ckpt_dir = Config.ckpt_dir
perf_dir = Config.perf_dir


def train(
    accelerator: str = trainer_config.accelerator,
    devices: int | str = trainer_config.devices,
    strategy: str = trainer_config.strategy,
    precision: str | None = trainer_config.precision,
    max_epochs: int = trainer_config.max_epochs,
    lr: float = module_config.learning_rate,
    batch_size: int = datamodule_config.batch_size,
    max_length: int = datamodule_config.max_length,
    deterministic: bool = trainer_config.deterministic,
    check_val_every_n_epoch: int | None = trainer_config.check_val_every_n_epoch,
    val_check_interval: int | float | None = trainer_config.val_check_interval,
    num_sanity_val_steps: int | None = trainer_config.num_sanity_val_steps,
    log_every_n_steps: int | None = trainer_config.log_every_n_steps,
    perf: bool = False,
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

    lit_module = SequenceClassificationModule(
        model_name=model_name,
        learning_rate=lr,
    )
    logger = CometLogger(
        api_key=os.getenv("COMET_API_KEY"),
        project="toxy-bot",
        workspace="anitamaxvim",
        name=experiment_name,
    )
    # logger = CSVLogger(save_dir=log_dir, version=version)

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
        logger=logger,
        callbacks=callbacks,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=log_every_n_steps,
        num_sanity_val_steps=num_sanity_val_steps,
        # enable_model_summary=False,
    )

    start = perf_counter()
    lit_trainer.fit(model=lit_module, datamodule=lit_datamodule)
    stop = perf_counter()

    if perf:
        log_perf(start, stop, lit_trainer, perf_dir, experiment_name)


if __name__ == "__main__":
    CLI(train, as_positional=False)  # type: ignore
