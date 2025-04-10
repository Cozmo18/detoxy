import lightning.pytorch as pl
from pathlib import Path
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG, TRAINER_CONFIG
from toxy_bot.ml.datamodule import tokenize_text



class SequenceClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = MODULE_CONFIG.model_name,
        num_labels: int = DATAMODULE_CONFIG.num_labels,
        learning_rate: float = MODULE_CONFIG.learning_rate,
        warmup_ratio: float | None = MODULE_CONFIG.warmup_ratio,
        output_key: str = "logits",
        loss_key: str = "loss",
        label_key: str = "labels",
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio

        self.output_key = output_key
        self.loss_key = loss_key
        self.label_key = label_key

        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="multi_label_classification"
        )
        
        # These will be set in setup()
        self.n_training_steps = None
        self.n_warmup_steps = None

        self.accuracy = MultilabelAccuracy(num_labels=self.num_labels)
        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)
        self.precision = MultilabelPrecision(num_labels=self.num_labels)
        self.recall = MultilabelRecall(num_labels=self.num_labels)

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            # Calculate total number of training steps
            num_training_samples = len(self.trainer.datamodule.train_data)
            num_epochs = self.trainer.max_epochs
            batch_size = self.trainer.datamodule.batch_size
            
            # Calculate steps per epoch
            steps_per_epoch = (num_training_samples + batch_size - 1) // batch_size  # Ceiling division
            
            # Total training steps
            self.n_training_steps = steps_per_epoch * num_epochs
            
            # Calculate warmup steps if warmup_ratio is specified
            if self.warmup_ratio is not None:
                self.n_warmup_steps = int(self.n_training_steps * self.warmup_ratio)
            else:
                self.n_warmup_steps = 0

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self.model(**batch)
        self.log("train_loss", outputs[self.loss_key], prog_bar=True)
        return outputs[self.loss_key]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.model(**batch)
        self.log("val_loss", outputs[self.loss_key], prog_bar=True)

        logits = outputs[self.output_key]
        labels = batch[self.label_key]

        acc = self.accuracy(logits, labels)
        f1 = self.f1_score(logits, labels)
        prec = self.precision(logits, labels)
        rec = self.recall(logits, labels)

        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_prec", prec, prog_bar=True)
        self.log("val_rec", rec, prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.model(**batch)

        logits = outputs[self.output_key]
        labels = batch[self.label_key]

        acc = self.accuracy(logits, labels)
        f1 = self.f1_score(logits, labels)
        prec = self.precision(logits, labels)
        rec = self.recall(logits, labels)

        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_prec", prec, prog_bar=True)
        self.log("test_rec", rec, prog_bar=True)

    def predict_step(
        self,
        sequence: str,
        cache_dir: str | Path = CONFIG.cache_dir,
        label_cols: list[str] = DATAMODULE_CONFIG.label_cols,
        max_token_len: int = DATAMODULE_CONFIG.max_token_len,
    ) -> torch.Tensor:
        batch = tokenize_text(
            sequence,
            model_name=self.model_name,
            cache_dir=cache_dir,
            max_length=max_token_len,
        )
        # Autotokenizer may cause tokens to lose device type and cause failure
        batch = batch.to(self.device)
        outputs = self.model(**batch)
        logits = outputs[self.output_key]
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
        return predictions

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        if self.warmup_ratio is not None and self.n_warmup_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.n_warmup_steps,
                num_training_steps=self.n_training_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
        else:
            return optimizer
