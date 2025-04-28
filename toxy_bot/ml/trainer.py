import os
from datetime import datetime

import lightning.pytorch as pl
import torch
from jsonargparse import CLI
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CometLogger

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG, TRAINER_CONFIG
from toxy_bot.ml.datamodule import AutoTokenizerDataModule
from toxy_bot.ml.module import SequenceClassificationModule
from toxy_bot.ml.utils import create_dirs

from dataclasses import asdict


# Constants
MODEL_NAME = MODULE_CONFIG.model_name

# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


def train(
    learning_rate: float = MODULE_CONFIG.learning_rate,
    max_seq_length: int = DATAMODULE_CONFIG.max_seq_length,
    batch_size: int = DATAMODULE_CONFIG.batch_size,
    cache_dir: str = CONFIG.cache_dir,
    log_dir: str = CONFIG.log_dir,
    ckpt_dir: str = CONFIG.ckpt_dir,
    fast_dev_run: bool = False,
) -> None:
    torch.set_float32_matmul_precision(precision="medium")

    create_dirs([log_dir, ckpt_dir])

    timestamp = datetime.now().strftime("%Y%m%d")
    experiment_name = f"{MODEL_NAME}__msl-{max_seq_length}__lr-{learning_rate}__bs-{batch_size}__{timestamp}"
    experiment_name = experiment_name.replace("/", "_")

    lit_datamodule = AutoTokenizerDataModule(
        model_name=MODEL_NAME,
        cache_dir=cache_dir,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )

    lit_model = SequenceClassificationModule(
        max_seq_length=max_seq_length, learning_rate=learning_rate, cache_dir=cache_dir,
    )

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        workspace=os.environ.get("COMET_WORKSPACE"),
        save_dir=log_dir,
        project_name="toxyy",
        mode="create",
        experiment_name=experiment_name,
    )
    comet_logger.log_hyperparams({"batch_size": batch_size})

    checkpoint_filename = experiment_name + "__{epoch}__{val_loss:.4f}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=checkpoint_filename,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min"),
        checkpoint_callback,
    ]
    
    trainer_config = asdict(TRAINER_CONFIG)
    lit_trainer = pl.Trainer(
        logger=comet_logger,
        callbacks=callbacks,
        fast_dev_run=fast_dev_run,
        **trainer_config,
    )

    lit_trainer.fit(model=lit_model, datamodule=lit_datamodule)

    if not fast_dev_run:
        lit_trainer.test(ckpt_path="best", datamodule=lit_datamodule)
        
def predict(ckpt_path: str):
    model = SequenceClassificationModule.load_from_checkpoint(ckpt_path)
    dm = AutoTokenizerDataModule.load_from_checkpoint(ckpt_path)
    
    trainer_config = asdict(TRAINER_CONFIG)
    lit_trainer = pl.Trainer(**trainer_config)
    
    predictions = lit_trainer.predict(model=model, datamodule=dm)
    
    return predictions
    
    

if __name__ == "__main__":
    CLI(train, as_positional=False)
