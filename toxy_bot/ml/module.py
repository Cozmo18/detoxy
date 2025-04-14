import lightning.pytorch as pl
from pathlib import Path
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG
from toxy_bot.ml.datamodule import tokenize_text



class SequenceClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = MODULE_CONFIG.model_name,
        num_labels: int = DATAMODULE_CONFIG.num_labels,
        max_token_len: int = DATAMODULE_CONFIG.max_token_len,
        input_key: str = "input_ids",
        labels_key: str = "labels",
        mask_key: str = "attention_mask",
        output_key: str = "logits",
        loss_key: str = "loss",
        learning_rate: float = MODULE_CONFIG.learning_rate,
        warmup_ratio: float | None = MODULE_CONFIG.warmup_ratio,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.max_token_len = max_token_len
        self.input_key = input_key
        self.labels_key = labels_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.loss_key = loss_key
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio

        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="multi_label_classification"
        )
        
        # These will be set in setup()
        self.n_training_steps = None
        self.n_warmup_steps = None

        self.auroc = MultilabelAUROC(num_labels=self.num_labels)
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

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            batch[self.input_key],
            attention_mask=batch[self.mask_key],
            labels=batch[self.label_key],
        )
        loss = outputs[self.loss_key]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, auroc, acc, f1, prec, rec = self._shared_eval_step(batch, batch_idx)
        metrics = {
            "val_loss": loss, 
            "val_auroc": auroc,
            "val_acc": acc, 
            "val_f1": f1, 
            "val_prec": prec, 
            "val_rec": rec,
        }
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics
    
    def test_step(self, batch, batch_idx):
        loss, auroc, acc, f1, prec, rec = self._shared_eval_step(batch, batch_idx)
        metrics = {
            "test_loss": loss, 
            "test_auroc": auroc,
            "test_acc": acc, 
            "test_f1": f1, 
            "test_prec": prec, 
            "test_rec": rec,
        }
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics
        
    def _shared_eval_step(self, batch, batch_idx):
        outputs = self.model(
            batch[self.input_key],
            attention_mask=batch[self.mask_key],
            labels=batch[self.label_key],
        )
        loss = outputs[self.loss_key]
        # If logits are provided, metrics still work
        auroc = self.auroc(outputs[self.output_key], batch[self.labels_key])
        acc = self.accuracy(outputs[self.output_key], batch[self.labels_key])
        f1 = self.f1_score(outputs[self.output_key], batch[self.labels_key])
        prec = self.precision(outputs[self.output_key], batch[self.labels_key])
        rec = self.recall(outputs[self.output_key], batch[self.labels_key])
        return loss, auroc, acc, f1, prec, rec
        
    def predict_step(self, sequence: str, cache_dir: str | Path = CONFIG.cache_dir):
        batch = tokenize_text(
            sequence,
            model_name=self.model_name,
            max_token_len=self.max_token_len,
            cache_dir=cache_dir,
        )
        # Autotokenizer may cause tokens to lose device type and cause failure
        batch = batch.to(self.device)
        outputs = self.model(**batch)
        logits = outputs[self.output_key]
        return torch.sigmoid(logits)

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
