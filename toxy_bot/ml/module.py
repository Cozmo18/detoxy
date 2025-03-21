import lightning.pytorch as pl

from toxy_bot.ml.config import DataModuleConfig, ModuleConfig


class SequenceClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = ModuleConfig.model_name,
        num_labels: int = DataModuleConfig.num_labels,  # set according to the finetuning dataset
        input_key: str = "input_ids",
        label_key: str = "label",
        mask_key: str = "attention_mask",
        output_key: str = "logits",
        loss_key: str = "loss",
        learning_rate: float = ModuleConfig.learning_rate,
    ) -> None:
        pass
