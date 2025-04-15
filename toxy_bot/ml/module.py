import lightning.pytorch as pl
from pathlib import Path
import torch
from torch.optim import AdamW
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
        learning_rate: float = MODULE_CONFIG.learning_rate,
        adam_epsilon: float = MODULE_CONFIG.adam_epsilon,
        warmup_steps: int = MODULE_CONFIG.warmup_steps,
        weight_decay: float = MODULE_CONFIG.weight_decay,
        cache_dir: str | Path = CONFIG.cache_dir,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.cache_dir = cache_dir

        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels, 
            problem_type="multi_label_classification",
            cache_dir=self.cache_dir,
        )
        
        self.auroc = MultilabelAUROC(num_labels=self.num_labels)
        self.accuracy = MultilabelAccuracy(num_labels=self.num_labels)
        self.f1_score = MultilabelF1Score(num_labels=self.num_labels)
        self.precision = MultilabelPrecision(num_labels=self.num_labels)
        self.recall = MultilabelRecall(num_labels=self.num_labels)
        
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # ON EPOCH?!
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
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        labels = batch["labels"]
        
        auroc = self.auroc(logits, labels.int())
        acc = self.accuracy(logits, labels)
        f1 = self.f1_score(logits, labels)
        prec = self.precision(logits, labels)
        rec = self.recall(logits, labels)
        
        return loss, auroc, acc, f1, prec, rec
        
    def predict_step(
        self, 
        sequence: str, 
        max_seq_length: int = DATAMODULE_CONFIG.max_seq_length,
        cache_dir: str | Path = CONFIG.cache_dir,
    ):
        batch = tokenize_text(
            sequence,
            model_name=self.model_name,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
        )
        # Autotokenizer may cause tokens to lose device type and cause failure
        batch = batch.to(self.device)
        outputs = self.model(**batch)
        probs = torch.sigmoid(outputs["logits"]).detach().cpu().numpy()
        return probs

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)."""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
