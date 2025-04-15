from time import perf_counter
import os
from datetime import datetime

import lightning.pytorch as pl
import torch
from jsonargparse import CLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG, TRAINER_CONFIG
from toxy_bot.ml.datamodule import AutoTokenizerDataModule
from toxy_bot.ml.module import SequenceClassificationModule
from toxy_bot.ml.utils import create_dirs, log_perf


# Constants
DATASET_NAME = DATAMODULE_CONFIG.dataset_name

def train(
    model_name = MODULE_CONFIG.model_name,
    lr: float = MODULE_CONFIG.learning_rate,
    adam_epsilon: float = MODULE_CONFIG.adam_epsilon,
    warmup_steps: int = MODULE_CONFIG.warmup_steps,
    weight_decay: float = MODULE_CONFIG.weight_decay,
    train_split: str = DATAMODULE_CONFIG.train_split,
    train_size: float = DATAMODULE_CONFIG.train_size,
    batch_size: int = DATAMODULE_CONFIG.batch_size,
    max_seq_length: int = DATAMODULE_CONFIG.max_seq_length,
    max_epochs: int = TRAINER_CONFIG.max_epochs,
    check_val_every_n_epoch: int | None = TRAINER_CONFIG.check_val_every_n_epoch,
    val_check_interval: int | float | None = TRAINER_CONFIG.val_check_interval,
    num_sanity_val_steps: int | None = TRAINER_CONFIG.num_sanity_val_steps,
    log_every_n_steps: int | None = TRAINER_CONFIG.log_every_n_steps,
    accelerator: str = TRAINER_CONFIG.accelerator,
    devices: int | str = TRAINER_CONFIG.devices,
    strategy: str = TRAINER_CONFIG.strategy,
    precision: str | None = TRAINER_CONFIG.precision,
    deterministic: bool = TRAINER_CONFIG.deterministic,
    cache_dir: str = CONFIG.cache_dir,
    log_dir: str = CONFIG.log_dir,
    ckpt_dir: str = CONFIG.ckpt_dir,
    perf_dir: str = CONFIG.perf_dir,
    perf: bool = False,
    fast_dev_run: bool = False,
) -> None:
    torch.set_float32_matmul_precision(precision="medium")
    
    create_dirs([log_dir, ckpt_dir, perf_dir])

    lit_datamodule = AutoTokenizerDataModule(
        dataset_name=DATASET_NAME,
        train_split=train_split,
        model_name=model_name,
        cache_dir=cache_dir,
        batch_size=batch_size,
        train_size=train_size,
        max_seq_length=max_seq_length,
    )

    lit_model = SequenceClassificationModule(
        model_name=model_name,
        learning_rate=lr,
        adam_epsilon=adam_epsilon,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )
    
    date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") 
    experiment_id = f"{model_name}_finetuned_{date}"
    
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        workspace=os.environ.get("COMET_WORKSPACE"),
        project="toxy-bot",
        mode="create",
        name=experiment_id,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=experiment_id,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,   
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if perf:
        callbacks = [checkpoint_callback, lr_monitor]
        num_sanity_val_steps = 0
    else:
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
            checkpoint_callback,
            lr_monitor,
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

    # Time training
    start = perf_counter()
    lit_trainer.fit(model=lit_model, datamodule=lit_datamodule)
    stop = perf_counter()

    if perf:
        log_perf(start, stop, lit_trainer, perf_dir, experiment_id)
        
    if not fast_dev_run:
        lit_trainer.test(ckpt_path="best", datamodule=lit_datamodule)


if __name__ == "__main__":
    CLI(train, as_positional=False)
