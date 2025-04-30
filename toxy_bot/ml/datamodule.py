import os
from datetime import datetime

import lightning.pytorch as pl
import numpy as np
from datasets import load_dataset
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from pathlib import Path

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG


class AutoTokenizerDataModule(pl.LightningDataModule):
    # Set according to the tokenizer output object
    loader_columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]

    def __init__(
        self,
        dataset_name: str = DATAMODULE_CONFIG.dataset_name,
        model_name: str = MODULE_CONFIG.model_name,
        labels: list[str] = DATAMODULE_CONFIG.labels,
        train_split: str = DATAMODULE_CONFIG.train_split,
        train_size: float = DATAMODULE_CONFIG.train_size,
        stratify_by_column: str = DATAMODULE_CONFIG.stratify_by_column,
        test_split: str = DATAMODULE_CONFIG.test_split,
        max_seq_length: int = DATAMODULE_CONFIG.max_seq_length,
        batch_size: int = DATAMODULE_CONFIG.batch_size,
        num_workers: int = DATAMODULE_CONFIG.num_workers,
        cache_dir: str | Path = CONFIG.cache_dir,
        seed: int = CONFIG.seed,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.labels = labels
        self.train_split = train_split
        self.train_size = train_size
        self.stratify_by_column = stratify_by_column
        self.test_split = test_split
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.seed = seed

    def prepare_data(self) -> None:
        pl.seed_everything(seed=self.seed)

        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        cache_dir_is_empty = len(os.listdir(self.cache_dir)) == 0

        # TODO: Check if dataset exists, not if empty

        if cache_dir_is_empty:
            rank_zero_info(f"[{str(datetime.now())}] Downloading dataset.")
            load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        else:
            rank_zero_info(
                f"[{str(datetime.now())}] Data cache exist. Loading from cache."
            )

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            # Load and split training data
            dataset = load_dataset(
                self.dataset_name, split=self.train_split, cache_dir=self.cache_dir
            )
            dataset = dataset.train_test_split(
                train_size=self.train_size, stratify_by_column=self.stratify_by_column
            )

            # Prep train
            self.train_data = dataset["train"].map(
                self.preprocess_data,
                batched=True,
                remove_columns=dataset["train"].column_names,
            )
            self.train_data.set_format("torch", columns=self.loader_columns)

            # Prep val
            self.val_data = dataset["test"].map(
                self.preprocess_data,
                batched=True,
                remove_columns=dataset["train"].column_names,
            )
            self.val_data.set_format("torch", columns=self.loader_columns)

            # Free memory from unneeded dataset obj
            del dataset

        if stage == "test" or stage is None:
            self.test_data = load_dataset(
                self.dataset_name, split=self.test_split, cache_dir=self.cache_dir
            )
            self.test_data = self.test_data.map(
                self.preprocess_data,
                batched=True,
                remove_columns=self.test_data.column_names,
            )
            self.test_data.set_format("torch", columns=self.loader_columns)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def preprocess_data(self, examples: dict):
        # Assume text col is "text" and tokenize
        encoding = tokenize_text(
            examples,
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            cache_dir=self.cache_dir,
        )
        # combine labels
        encoding["labels"] = combine_labels(examples, self.labels)

        return encoding


def tokenize_text(
    batch: str | dict,
    *,
    model_name: str,
    max_seq_length: int,
    cache_dir: str | Path,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, cache_dir=cache_dir
    )
    text = (
        batch if isinstance(batch, str) else batch["text"]
    )  # Allow for inference input as raw text
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )


def combine_labels(batch: dict, labels: list[str]) -> list:
    batch_size = len(batch[labels[0]])
    num_labels = len(labels)

    labels_batch = {k: batch[k] for k in batch.keys() if k in labels}
    labels_matrix = np.zeros((batch_size, num_labels))

    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    return labels_matrix.tolist()


if __name__ == "__main__":
    # Test the AutoTokenizerDataModule
    print("Testing AutoTokenizerDataModule...")

    # Initialize the datamodule with test parameters
    dm = AutoTokenizerDataModule(
        batch_size=8,
        max_seq_length=128,
        train_size=0.8,
    )

    # Test prepare_data
    print("Testing prepare_data...")
    dm.prepare_data()

    # Test setup
    print("Testing setup...")
    dm.setup("fit")

    # Test dataloaders
    print("Testing dataloaders...")
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    # Print some basic information
    print(f"Number of training batches: {len(train_dl)}")
    print(f"Number of validation batches: {len(val_dl)}")

    # Test a single batch
    print("\nTesting a single batch...")
    batch = next(iter(train_dl))
    print(batch)
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")

    print("\nTest completed successfully!")
