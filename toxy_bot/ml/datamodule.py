import os

import lightning.pytorch as pl
import numpy as np
from datasets import load_dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG


class AutoTokenizerDataModule(pl.LightningDataModule):
    loader_columns = ["input_ids", "attention_mask", "labels"]

    def __init__(
        self,
        dataset_name: str = DATAMODULE_CONFIG.dataset_name,
        model_name: str = MODULE_CONFIG.model_name,
        labels: list[str] = DATAMODULE_CONFIG.labels,
        batch_size: int = DATAMODULE_CONFIG.batch_size,
        max_seq_length: int = DATAMODULE_CONFIG.max_seq_length,
        train_split: str = DATAMODULE_CONFIG.train_split,
        train_size: float = DATAMODULE_CONFIG.train_size,
        stratify_by_column: str = DATAMODULE_CONFIG.stratify_by_column,
        test_split: str = DATAMODULE_CONFIG.test_split,
        num_workers: int = DATAMODULE_CONFIG.num_workers,
        cache_dir: str = CONFIG.cache_dir,
        seed: int = CONFIG.seed,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.labels = labels
        self.num_labels = len(labels)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.train_split = train_split
        self.test_split = test_split
        self.train_size = train_size
        self.stratify_by_column = stratify_by_column
        self.num_workers = num_workers
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=True
        )

    def prepare_data(self) -> None:
        pl.seed_everything(seed=self.seed)

        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, use_fast=True
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
            self.train_data.set_format("torch")

            # Prep val
            self.val_data = dataset["test"].map(
                self.preprocess_data,
                batched=True,
                remove_columns=dataset["train"].column_names,
            )
            self.val_data.set_format("torch")

            # Free memory from unneeded dataset obj
            del dataset

        if stage == "test" or stage == "predict" or stage is None:
            self.test_data = load_dataset(
                self.dataset_name, split=self.test_split, cache_dir=self.cache_dir
            )
            self.test_data = self.test_data.map(
                self.preprocess_data,
                batched=True,
                remove_columns=self.test_data.column_names,
            )
            self.test_data.set_format("torch")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
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
        # take a batch of texts, assume column name is "text"
        text = examples["text"]
        # encode them
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_seq_length
        )
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding


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
