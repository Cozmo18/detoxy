from time import perf_counter

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from toxy_bot.ml.config import Config, DataModuleConfig, ModuleConfig, TrainerConfig
from toxy_bot.ml.datamodule import AutoTokenizerDataModule
from toxy_bot.ml.module import SequenceClassificationModule
from toxy_bot.ml.utils import create_dirs, log_perf

# Config instances
config = Config()
datamodule_config = DataModuleConfig()
module_config = ModuleConfig()
trainer_config = TrainerConfig()

# Constants
MODEL_NAME: str = module_config.model_name
DATASET_NAME: str = datamodule_config.dataset_name

# Paths
CACHE_DIR: str = config.cache_dir
LOG_DIR: str = config.log_dir
CKPT_DIR: str = config.ckpt_dir
PERF_DIR: str = config.perf_dir

create_dirs(dirs=[CACHE_DIR, LOG_DIR, CKPT_DIR, PERF_DIR])

torch.set_float32_matmul_precision(precision="medium")


def train(
    accelerator: str = trainer_config.accelerator,
    devices: int | str = trainer_config.devices,
    strategy: str = trainer_config.strategy,
    precision: str | None = trainer_config.precision,
    max_epochs: int = trainer_config.max_epochs,
    lr: float = module_config.learning_rate,
    batch_size: int = datamodule_config.batch_size,
    deterministic: bool = trainer_config.deterministic,
    log_every_n_steps: int | None = trainer_config.log_every_n_steps,
    num_sanity_val_steps: int | None = trainer_config.num_sanity_val_steps,
    perf: bool = False,
) -> None:
    lit_datamodule = AutoTokenizerDataModule(
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        cache_dir=CACHE_DIR,
        batch_size=batch_size,
    )

    lit_module = SequenceClassificationModule(learning_rate=lr)

    logger = CSVLogger(save_dir=LOG_DIR, name="csv_logs")

    # do not use EarlyStopping if getting perf benchmark
    if perf:
        callbacks = [
            ModelCheckpoint(dirpath=CKPT_DIR, filename="model"),
        ]
        # Force num_sanity_val_steps to None for accurate perf benchmarking
        num_sanity_val_steps = None
    else:
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(dirpath=CKPT_DIR, filename="model"),
        ]

    lit_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        deterministic=deterministic,
        logger=logger,
        callbacks=callbacks,  # type: ignore
        log_every_n_steps=log_every_n_steps,
        num_sanity_val_steps=num_sanity_val_steps,
    )

    start = perf_counter()
    lit_trainer.fit(model=lit_module, datamodule=lit_datamodule)
    stop = perf_counter()

    if perf:
        log_perf(start, stop, PERF_DIR, lit_trainer)

    lit_trainer.fit(model=lit_module, datamodule=lit_datamodule)


if __name__ == "__main__":
    train(perf=True)
