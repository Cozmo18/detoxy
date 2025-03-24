import os
from datetime import datetime
from typing import Any, Optional

import lightning.pytorch as pl
from datasets import Dataset, load_dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from toxy_bot.ml.config import Config, DataModuleConfig, ModuleConfig


class AutoTokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = DataModuleConfig.dataset_name,
        data_dir: Optional[str] = Config.external_dir,
        cache_dir: str = Config.cache_dir,
        text_col: str = DataModuleConfig.text_col,
        label_cols: list[str] = DataModuleConfig().label_cols,
        num_labels: int = DataModuleConfig.num_labels,
        columns: list[str] = ["input_ids", "attention_mask", "labels"],
        model_name: str = ModuleConfig.model_name,
        batch_size: int = DataModuleConfig.batch_size,
        max_length: int = DataModuleConfig.max_length,
        train_split: str = DataModuleConfig.train_split,
        test_split: str = DataModuleConfig.test_split,
        train_size: float = DataModuleConfig.train_size,
        num_workers: int = DataModuleConfig.num_workers,
        seed: int = Config.seed,
    ) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.data_dir = data_dir
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

    def prepare_data(self) -> None:
        pl.seed_everything(seed=self.seed)

        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not os.path.isdir(s=self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        cache_dir_is_empty: bool = len(os.listdir(self.cache_dir)) == 0

        if cache_dir_is_empty:
            rank_zero_info(f"[{str(datetime.now())}] Downloading dataset.")
            load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                data_dir=self.data_dir,
                trust_remote_code=True,
            )
        else:
            rank_zero_info(
                f"[{str(datetime.now())}] Data cache exists. Loading from cache."
            )

    def setup(self, stage: str) -> None:
        dataset = None

        if stage == "fit" or stage is None:
            # Load and split training data
            dataset = load_dataset(
                self.dataset_name,
                split=self.train_split,
                cache_dir=self.cache_dir,
                data_dir=self.data_dir,
            )
            dataset = dataset.train_test_split(train_size=self.train_size)  # type: ignore

            # Prep train
            self.train_data = preprocess(
                dataset["train"],
                self.text_col,
                self.label_cols,
                self.model_name,
                self.cache_dir,
                self.max_length,
                self.columns,
            )

            # Prep val
            self.val_data = preprocess(
                dataset["test"],
                self.text_col,
                self.label_cols,
                self.model_name,
                self.cache_dir,
                self.max_length,
                self.columns,
            )

        if stage == "test" or stage is None:
            dataset = load_dataset(
                self.dataset_name,
                split=self.test_split,
                cache_dir=self.cache_dir,
                data_dir=self.data_dir,
            )
            self.test_data = preprocess(
                dataset,
                self.text_col,
                self.label_cols,
                self.model_name,
                self.cache_dir,
                self.max_length,
                self.columns,
            )

        # Free memory from unneeded dataset obj
        if dataset is not None:
            del dataset

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


def preprocess(
    dataset: Dataset,
    text_col: str,
    label_cols: list[str],
    model_name: str,
    cache_dir: str,
    max_length: int,
    cols_to_keep: list[str],
) -> Dataset:
    # Combine labels and convert to float
    prepared_dataset = dataset.map(
        lambda example: {"labels": [float(example[label]) for label in label_cols]},
        batched=False,
    )

    # Tokenize inputs
    prepared_dataset = prepared_dataset.map(
        function=tokenize_text,
        batched=True,
        batch_size=None,
        fn_kwargs={
            "text_col": text_col,
            "model_name": model_name,
            "cache_dir": cache_dir,
            "max_length": max_length,
        },
    )

    # Set format for PyTorch
    prepared_dataset.set_format(type="torch", columns=cols_to_keep)

    return prepared_dataset


def tokenize_text(
    batch: Any,  # TODO: fix type
    *,
    model_name: str,
    cache_dir: str,
    max_length: int,
    text_col: Optional[str] = None,
    # add_special_tokens: bool = True,
    truncation: bool = True,
    padding: str = "max_length",
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    text = batch if isinstance(batch, str) else batch[text_col]

    return tokenizer(
        text,
        # add_special_tokens=add_special_tokens,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors="pt",
    )
