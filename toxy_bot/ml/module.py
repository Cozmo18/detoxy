from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG


class SequenceClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = MODULE_CONFIG.model_name,
        label_columns: list[str] = DATAMODULE_CONFIG.labels,
        max_seq_length: int = DATAMODULE_CONFIG.max_seq_length,
        learning_rate: float = MODULE_CONFIG.learning_rate,
        cache_dir: str | Path = CONFIG.cache_dir,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model_name = model_name
        self.label_columns = label_columns
        self.num_labels = len(label_columns)
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.cache_dir = cache_dir
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=self.num_labels, problem_type="multi_label_classification", cache_dir=self.cache_dir
        )

        self.accuracy = MultilabelAccuracy(num_labels=self.num_labels)
        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)

    def forward(self, **inputs) -> torch.Tensor:
        return self.model(**inputs)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss, acc, f1 = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss, "val_acc": acc, "val_f1": f1}
        self.log_dict(metrics, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx) -> None:
        loss, acc, f1 = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss, "test_acc": acc, "test_f1": f1}
        self.log_dict(metrics, prog_bar=True, logger=True)

    def _shared_eval_step(self, batch, batch_idx) -> tuple:
        outputs = self.model(**batch)
        labels = batch["labels"]

        loss, logits = outputs[:2]
        acc = self.accuracy(logits, labels)
        f1 = self.f1_score(logits, labels)

        return loss, acc, f1

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        if isinstance(batch, str):
            encoding = self.tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_seq_length)
            encoding = encoding.to(self.device)
        else:
            encoding = batch
            
        outputs = self.model(**encoding)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
        return probabilities

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

