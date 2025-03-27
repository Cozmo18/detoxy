import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from transformers import BertForSequenceClassification

from toxy_bot.ml.config import Config, DataModuleConfig, ModuleConfig

# Create instances of config classes
config = Config()
datamodule_config = DataModuleConfig()
module_config = ModuleConfig()


class SequenceClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = module_config.model_name,
        num_labels: int = datamodule_config.num_labels,
        output_key: str = "logits",
        loss_key: str = "loss",
        learning_rate: float = module_config.learning_rate,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.output_key = output_key
        self.loss_key = loss_key
        self.learning_rate = learning_rate

        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="multi_label_classification"
        )

        self.accuracy = MultilabelAccuracy(num_labels=self.num_labels)
        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)
        self.precision = MultilabelPrecision(num_labels=self.num_labels)
        self.recall = MultilabelRecall(num_labels=self.num_labels)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        outputs = self.model(**batch)
        self.log("train_loss", outputs[self.loss_key], prog_bar=True)
        return outputs[self.loss_key]

    def validation_step(self, batch, batch_idx) -> None:
        outputs = self.model(**batch)
        self.log("val_loss", outputs[self.loss_key], prog_bar=True)

        logits = outputs[self.output_key]
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()

        acc = self.accuracy(predictions, batch["labels"])
        f1 = self.f1_score(predictions, batch["labels"])
        prec = self.precision(predictions, batch["labels"])
        rec = self.recall(predictions, batch["labels"])

        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_prec", prec, prog_bar=True)
        self.log("val_rec", rec, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        outputs = self.model(**batch)
        logits = outputs[self.output_key]
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()

        acc = self.accuracy(predictions, batch["labels"])
        f1 = self.f1_score(predictions, batch["labels"])
        prec = self.precision(predictions, batch["labels"])
        rec = self.recall(predictions, batch["labels"])

        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_prec", prec, prog_bar=True)
        self.log("test_rec", rec, prog_bar=True)

    # def predict_step(
    #     self, text: str, cache_dir: str = Config.cache_dir
    # ) -> torch.Tensor:
    #     batch = tokenize_text(
    #         batch=sequence,
    #         model_name=self.model_name,
    #         max_length=self.max_length,
    #         cache_dir=cache_dir,
    #     )
    #     batch = batch.to(self.device)
    #     outputs = self.model(**batch)
    #     logits = outputs[self.output_key]
    #     probabilities = torch.sigmoid(logits)
    #     predictions = (probabilities > 0.5).float()
    #     return predictions

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.learning_rate)
        return optimizer
