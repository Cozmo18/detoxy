from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW
from torchmetrics.classification import MultilabelAccuracy

# from torch.nn.functional import binary_cross_entropy_wiA
from model.config import Config, DataModuleConfig, ModuleConfig
from model.utils import get_model_and_tokenizer


class SequenceClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = ModuleConfig.model_name,
        num_classes: int = len(DataModuleConfig.label_cols),
        max_token_len: int = ModuleConfig.max_token_len,
        learning_rate: float = ModuleConfig.learning_rate,
        cache_dir: str | Path = Config.cache_dir,
        input_key: str = "input_ids",
        mask_key: str = "attention_mask",
        output_key: str = "logits",
        loss_key: str = "loss",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_token_len = max_token_len
        self.learning_rate = learning_rate
        self.ache_dir = cache_dir
        self.input_key = input_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.loss_key = loss_key

        self.tokenizer, self.model = get_model_and_tokenizer(
            model_name, num_labels=num_classes, cache_dir=cache_dir
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.accuracy = MultilabelAccuracy(num_labels=num_classes)

    def forward(self, x: str | list[str]) -> torch.Tensor:
        encodings = self.tokenizer(
            x,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        outputs = self.model(
            encodings[self.input_key],
            attention_mask=encodings[self.mask_key],
        )
        return outputs[self.output_key]

    def training_step(
        self, batch: tuple[list[str], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        text, labels = batch
        logits = self.forward(text)
        loss = self.criterion(logits, labels)
        self.log("train-loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[list[str], torch.Tensor], batch_idx: int
    ) -> None:
        text, labels = batch
        logits = self.forward(text)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        acc = self.accuracy(preds, labels)
        self.log("val-loss", loss, prog_bar=True)
        self.log("val-acc", acc, prog_bar=True)

    def test_step(self, batch: tuple[list[str], torch.Tensor], batch_idx: int) -> None:
        text, labels = batch
        logits = self.forward(text)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        acc = self.accuracy(preds, labels)
        self.log("test-loss", loss, prog_bar=True)
        self.log("test-acc", acc, prog_bar=True)

    def predict_step(self, text: str) -> torch.Tensor:
        logits = self.forward(text)
        preds = torch.sigmoid(logits)
        return preds.flatten()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return AdamW(self.parameters(), lr=self.learning_rate)
