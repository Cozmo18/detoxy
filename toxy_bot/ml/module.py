import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import Metric
from transformers import BertForSequenceClassification

from toxy_bot.ml.config import DataModuleConfig, ModuleConfig
from toxy_bot.ml.datamodule import tokenize_text


class SequenceClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = ModuleConfig.model_name,
        num_labels: int = DataModuleConfig.num_labels,
        input_key: str = "input_ids",
        label_key: str = "label",
        mask_key: str = "attention_mask",
        output_key: str = "logits",
        loss_key: str = "loss",
        learning_rate: float = ModuleConfig.learning_rate,
        accuracy: Metric = ModuleConfig.accuracy,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.input_key = input_key
        self.label_key = label_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.loss_key = loss_key
        self.learning_rate = learning_rate
        self.accuracy = accuracy

        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            # num_labels=num_labels,
        )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        outputs = self.model(
            batch[self.input_key],
            attention_mask=batch[self.mask_key],
            labels=batch[self.label_key],
        )
        self.log(name="train_loss", value=outputs[self.loss_key])
        return outputs[self.loss_key]

    def validation_step(self, batch, batch_idx) -> None:
        outputs = self.model(
            batch[self.input_key],
            attention_mask=batch[self.mask_key],
            labels=batch[self.label_key],
        )
        self.log(name="val_loss", value=outputs[self.loss_key], prog_bar=True)

        # logits = outputs[self.output_key]
        # # TODO: Check if need to use sigmoid
        # acc = self.accuracy.update(logits, batch[self.label_key]).compute()
        # self.log(name="val_acc", value=acc, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        outputs = self.model(
            batch[self.input_key],
            attention_mask=batch[self.mask_key],
            labels=batch[self.label_key],
        )
        # logits = outputs[self.output_key]
        # TODO: Check if need to use sigmoid
        # acc = self.accuracy.update(logits, batch[self.label_key]).compute()
        # self.log(name="test_acc", value=acc, prog_bar=True)

    def predict_step(self, sequence: str) -> torch.Tensor:
        batch = tokenize_text(
            batch=sequence,
            model_name=self.model_name,
            max_length=self.max_length,
        )
        batch = batch.to(self.device)
        outputs = self.model(batch[self.input_key])
        logits = outputs[self.output_key]
        # TODO: Get actual labels
        predicted_labels = (torch.sigmoid(logits) > 0.5).float()
        return predicted_labels

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        return optimizer
