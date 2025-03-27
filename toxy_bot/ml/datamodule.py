import os
from datetime import datetime

import lightning.pytorch as pl
from datasets import load_dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from toxy_bot.ml.config import Config, DataModuleConfig, ModuleConfig

# Create instances of config classes
config = Config()
datamodule_config = DataModuleConfig()
module_config = ModuleConfig()


class AutoTokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = datamodule_config.dataset_name,
        cache_dir: str = config.cache_dir,
        text_col: str = datamodule_config.text_col,
        label_cols: list[str] = datamodule_config.label_cols,
        num_labels: int = datamodule_config.num_labels,
        columns: list[str] = ["input_ids", "attention_mask", "labels"],
        model_name: str = module_config.model_name,
        batch_size: int = datamodule_config.batch_size,
        max_length: int = datamodule_config.max_length,
        train_split: str = datamodule_config.train_split,
        test_split: str = datamodule_config.test_split,
        train_size: float = datamodule_config.train_size,
        num_workers: int = datamodule_config.num_workers,
        seed: int = config.seed,
    ) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.text_col = text_col
        self.label_cols = label_cols
        self.num_lables = num_labels
        self.columns = columns
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_split = train_split
        self.test_split = test_split
        self.train_size = train_size
        self.num_workers = num_workers
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=True
        )

        self.format = {
            "type": "torch",
            # "format_kwargs": {"dtype": torch.float},
            "columns": self.columns,
        }

    def prepare_data(self) -> None:
        pl.seed_everything(seed=self.seed)

        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not os.path.isdir(s=self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        cached_dir_dataset = os.path.join(
            self.cache_dir, self.dataset_name.replace("/", "___")
        )
        dataset_cached = os.path.exists(cached_dir_dataset)

        if not dataset_cached:
            rank_zero_info(
                f"[{str(datetime.now())}] Downloading dataset {self.dataset_name}."
            )
            load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        else:
            rank_zero_info(
                f"[{str(datetime.now())}] Dataset {self.dataset_name} exists in cache. Loading from cache."
            )

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            # Load and split training data
            dataset = load_dataset(
                self.dataset_name, split=self.train_split, cache_dir=self.cache_dir
            )
            dataset = dataset.train_test_split(train_size=self.train_size)  # type: ignore

            self.train_data = dataset["train"].map(
                self.preprocess,
                batched=True,
                batch_size=self.batch_size,
            )
            self.train_data.set_format(**self.format)

            self.val_data = dataset["test"].map(
                self.preprocess,
                batched=True,
                batch_size=self.batch_size,
            )
            self.val_data.set_format(**self.format)

            del dataset

        if stage == "test" or stage is None:
            self.test_data = load_dataset(
                self.dataset_name, split=self.test_split, cache_dir=self.cache_dir
            )
            self.test_data.map(
                self.preprocess,
                batched=True,
                batch_size=self.batch_size,
            )
            self.test_data.set_format(**self.format)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def preprocess(self, batch: str | dict) -> dict:
        if isinstance(batch, str):
            return self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            # Tokenize the text
            tokenized = self.tokenizer(
                batch[self.text_col],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,  # Don't return tensors yet. Use set_format(type="torch") instead.
            )

            # Create a combined labels column
            labels = []
            for i in range(len(batch[self.text_col])):
                row_labels = [float(batch[col][i]) for col in self.label_cols]
                labels.append(row_labels)

            tokenized["labels"] = labels
            return tokenized


if __name__ == "__main__":
    dm = AutoTokenizerDataModule()
    dm.prepare_data()
    dm.setup(stage="fit")
    print(dm.train_data[0])
