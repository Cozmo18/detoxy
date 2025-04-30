from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score
from transformers import AutoModelForSequenceClassification

from toxy_bot.ml.config import DATAMODULE_CONFIG, MODULE_CONFIG, CONFIG
from toxy_bot.ml.datamodule import tokenize_text


class SequenceClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = MODULE_CONFIG.model_name,
        labels: list[str] = DATAMODULE_CONFIG.labels,
        max_seq_length: int = DATAMODULE_CONFIG.max_seq_length,
        learning_rate: float = MODULE_CONFIG.learning_rate,
        output_key: str = "logits",  # Set according to the model output object
        loss_key: str = "loss",  # Set according to the model output object
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model_name = model_name
        self.labels = labels
        self.num_labels = len(labels)
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.output_key = output_key
        self.loss_key = loss_key

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            problem_type="multi_label_classification",
            num_labels=self.num_labels,
        )

        self.accuracy = MultilabelAccuracy(num_labels=self.num_labels)
        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[self.loss_key]
        self.log("train-loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, acc, f1 = self._shared_eval_step(batch, batch_idx)
        metrics = {"val-loss": loss, "val-acc": acc, "val-f1": f1}
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, acc, f1 = self._shared_eval_step(batch, batch_idx)
        metrics = {"test-loss": loss, "test-acc": acc, "test-f1": f1}
        self.log_dict(metrics, prog_bar=True)

    def _shared_eval_step(self, batch: dict, batch_idx: int) -> tuple:
        labels = batch["labels"]
        outputs = self.model(**batch)
        loss = outputs[self.loss_key]
        logits = outputs[self.output_key]
        acc = self.accuracy(logits, labels)  # accept logits as pred
        f1 = self.f1_score(logits, labels)  # accept logits as pred

        return loss, acc, f1

    def predict_step(
        self,
        batch: str,
        cache_dir: str | Path = CONFIG.cache_dir,
    ) -> torch.Tensor:
        encoding = tokenize_text(
            batch,
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            cache_dir=cache_dir,
        )
        encoding = encoding.to(self.device)
        outputs = self.model(**encoding)
        logits = outputs[self.output_key]
        probabilities = torch.sigmoid(logits).flatten()
        probabilities = probabilities.cpu().detech().numpy()

        return {{label: prob for label, prob in zip(self.labels, probabilities)}}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
