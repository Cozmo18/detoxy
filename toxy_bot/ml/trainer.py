from time import perf_counter

import lightning.pytorch as pl
import torch
from jsonargparse import CLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from toxy_bot.ml.config import Config, DataModuleConfig, ModuleConfig, TrainerConfig
from toxy_bot.ml.datamodule import AutoTokenizerDataModule
from toxy_bot.ml.module import SequenceClassificationModule
from toxy_bot.ml.utils import create_dirs, create_run_name, log_perf

# Config instances
config = Config()
datamodule_config = DataModuleConfig()
module_config = ModuleConfig()
trainer_config = TrainerConfig()


torch.set_float32_matmul_precision(precision="medium")


def train(
    model_name: str = module_config.model_name,
    accelerator: str = trainer_config.accelerator,
    devices: int | str = trainer_config.devices,
    strategy: str = trainer_config.strategy,
    precision: str | None = trainer_config.precision,
    max_epochs: int = trainer_config.max_epochs,
    lr: float = module_config.learning_rate,
    batch_size: int = datamodule_config.batch_size,
    max_length: int = datamodule_config.max_length,
    deterministic: bool = trainer_config.deterministic,
    log_every_n_steps: int | None = trainer_config.log_every_n_steps,
    num_sanity_val_steps: int | None = trainer_config.num_sanity_val_steps,
    perf: bool = False,
    log_dir: str = config.log_dir,
    ckpt_dir: str = config.ckpt_dir,
    perf_dir: str = config.perf_dir,
) -> None:
    # Create required directories
    create_dirs([log_dir, ckpt_dir, perf_dir])

    # Create unique run name
    run_name = create_run_name(
        model_name=model_name,
        learning_rate=lr,
        batch_size=batch_size,
        max_length=max_length,
    )

    lit_datamodule = AutoTokenizerDataModule(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )

    lit_model = SequenceClassificationModule(
        model_name=model_name,
        learning_rate=lr,
    )

    logger = CSVLogger(save_dir=log_dir, version=run_name)

    # do not use EarlyStopping if getting perf benchmark
    # do not perform sanity checking if getting perf benchmark
    if perf:
        callbacks = [
            ModelCheckpoint(dirpath=ckpt_dir, filename=run_name),
        ]
        num_sanity_val_steps = 0
    else:
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(dirpath=ckpt_dir, filename=run_name),
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
        log_every_n_steps=log_every_n_steps,
        num_sanity_val_steps=num_sanity_val_steps,
    )

    start = perf_counter()
    lit_trainer.fit(model=lit_model, datamodule=lit_datamodule)
    stop = perf_counter()

    if perf:
        log_perf(start, stop, perf_dir, lit_trainer, run_name)


if __name__ == "__main__":
    CLI(train, as_positional=False)
