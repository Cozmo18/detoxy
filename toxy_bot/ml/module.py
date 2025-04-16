import lightning.pytorch as pl
from pathlib import Path
import torch
from torch.optim import AdamW
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score
from transformers import BertForSequenceClassification, get_inverse_sqrt_schedule

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG


class SequenceClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = MODULE_CONFIG.model_name,
        num_labels: int = DATAMODULE_CONFIG.num_labels,
        learning_rate: float = MODULE_CONFIG.learning_rate,
        adam_epsilon: float = MODULE_CONFIG.adam_epsilon,
        warmup_ratio: float = MODULE_CONFIG.warmup_ratio,
        weight_decay: float = MODULE_CONFIG.weight_decay,
        cache_dir: str | Path = CONFIG.cache_dir,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.cache_dir = cache_dir

        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels, 
            problem_type="multi_label_classification",
            cache_dir=self.cache_dir,
        )
        
        self.accuracy = MultilabelAccuracy(num_labels=self.num_labels)
        self.f1_score = MultilabelF1Score(num_labels=num_labels)
        
    def setup(self, stage):
        if stage == "fit":
             num_training_samples = len(self.trainer.datamodule.dataset["train"])
             num_epochs = self.trainer.max_epochs
             batch_size = self.trainer.datamodule.batch_size
             
             steps_per_epoch = (num_training_samples + batch_size - 1) // batch_size  # Ceiling division
             num_training_steps = steps_per_epoch * num_epochs
             
             if self.warmup_ratio > 0:
                 self.num_warmup_steps = int(num_training_steps * self.warmup_ratio)
             else:
                 self.num_warmup_steps = 0
        
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # ON EPOCH?!
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc, f1 = self._shared_eval_step(batch, batch_idx)
        metrics = { "val_loss": loss, "val_acc": acc, "val_f1": f1 }
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics
    
    def test_step(self, batch, batch_idx):
        loss, acc, f1 = self._shared_eval_step(batch, batch_idx)
        metrics = { "test_loss": loss, "test_acc": acc, "test_f1": f1 }
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics
        
    def _shared_eval_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        labels = batch["labels"] 
        
        loss, logits = outputs[:2]
        acc = self.accuracy(logits, labels)
        f1 = self.f1_score(logits, labels)
        
        return loss, acc, f1
        
    # def predict_step(
    #     self, 
    #     sequence: str, 
    #     max_seq_length: int = DATAMODULE_CONFIG.max_seq_length,
    #     cache_dir: str | Path = CONFIG.cache_dir,
    # ):
    #     batch = tokenize_text(
    #         sequence,
    #         model_name=self.model_name,
    #         cache_dir=cache_dir,
    #         max_seq_length=max_seq_length,
    #     )
    #     # Autotokenizer may cause tokens to lose device type and cause failure
    #     batch = batch.to(self.device)
    #     outputs = self.model(**batch)
    #     probs = torch.sigmoid(outputs["logits"]).detach().cpu().numpy()
    #     return probs
    
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=self.num_warmup_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
        

    # def configure_optimizers(self):
    #     """Prepare optimizer and schedule (linear warmup and decay)."""
    #     model = self.model
    #     no_decay = ["bias", "LayerNorm.weight"]
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #             "weight_decay": self.hparams.weight_decay,
    #         },
    #         {
    #             "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #     optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=self.hparams.warmup_steps,
    #         num_training_steps=self.trainer.estimated_stepping_batches,
    #     )
    #     scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    #     return [optimizer], [scheduler]
