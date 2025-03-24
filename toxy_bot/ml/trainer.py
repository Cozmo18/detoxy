from typing import Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from toxy_bot.ml.config import Config, DataModuleConfig, ModuleConfig, TrainerConfig
from toxy_bot.ml.datamodule import AutoTokenizerDataModule
from toxy_bot.ml.module import SequenceClassificationModule
from toxy_bot.ml.utils import create_dirs

# Constants
MODEL_NAME: str = ModuleConfig.model_name
DATASET_NAME: str = DataModuleConfig.dataset_name

# Paths
CACHE_DIR: str = Config.cache_dir
LOG_DIR: str = Config.log_dir
CKPT_DIR: str = Config.ckpt_dir

create_dirs(dirs=[CACHE_DIR, LOG_DIR, CKPT_DIR])

torch.set_float32_matmul_precision(precision="medium")


def train(
    accelerator: str = TrainerConfig.accelerator,
    devices: int | str = TrainerConfig.devices,
    strategy: str = TrainerConfig.strategy,
    precision: Optional[str] = TrainerConfig.precision,
    max_epochs: int = TrainerConfig.max_epochs,
    lr: float = ModuleConfig.learning_rate,
    batch_size: int = DataModuleConfig.batch_size,
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
    else:
        callbacks = [
            EarlyStopping(monitor="val_acc", mode="min", patience=3),
            ModelCheckpoint(dirpath=CKPT_DIR, filename="model"),
        ]

    lit_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,  # type: ignore
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,  # type: ignore
        log_every_n_steps=50,
    )

    lit_trainer.fit(model=lit_module, datamodule=lit_datamodule)


if __name__ == "__main__":
    train()
