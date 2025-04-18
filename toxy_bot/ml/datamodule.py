import os
from pathlib import Path

import lightning.pytorch as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG


class AutoTokenizerDataModule(pl.LightningDataModule):
    loader_columns = ["input_ids", "attention_mask", "labels"]

    def __init__(
        self,
        dataset_name: str = DATAMODULE_CONFIG.dataset_name,
        model_name: str = MODULE_CONFIG.model_name,
        text_column: str = DATAMODULE_CONFIG.text_column,
        label_columns: list[str] = DATAMODULE_CONFIG.label_columns,
        num_labels: int = DATAMODULE_CONFIG.num_labels,
        batch_size: int = DATAMODULE_CONFIG.batch_size,
        max_seq_length: int = DATAMODULE_CONFIG.max_seq_length,
        train_split: str = DATAMODULE_CONFIG.train_split,
        train_size: float = DATAMODULE_CONFIG.train_size,
        stratify_by_column: str = DATAMODULE_CONFIG.stratify_by_column,
        test_split: str = DATAMODULE_CONFIG.test_split,
        num_workers: int = DATAMODULE_CONFIG.num_workers,
        cache_dir: str | Path = CONFIG.cache_dir,
        seed: int = CONFIG.seed,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.text_column = text_column
        self.label_columns = label_columns
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.train_split = train_split
        self.test_split = test_split
        self.train_size = train_size
        self.stratify_by_column = stratify_by_column
        self.num_workers = num_workers
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, use_fast=True
        )

    def prepare_data(self):
        pl.seed_everything(seed=self.seed)

        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, use_fast=True
        )

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            # Load and split data
            dataset = load_dataset(self.dataset_name, split=self.train_split, cache_dir=self.cache_dir)
            dataset = dataset.train_test_split(train_size=self.train_size, stratify_by_column=self.stratify_by_column)
            
            self.train_data = dataset["train"].map(self.convert_to_features, batched=True)
            self.train_data.set_format(type="torch", columns=self.loader_columns)
            
            self.val_data = dataset["test"].map(self.convert_to_features, batched=True)
            self.val_data.set_format(type="torch", columns=self.loader_columns)
            
            del dataset
            
        if stage == "test" or stage is None:
            self.test_data = load_dataset(self.dataset_name, split=self.test_split, cache_dir=self.cache_dir)
            self.test_data = self.test_data.map(self.convert_to_features, batched=True)
            self.test_data.set_format(type="torch", columns=self.loader_columns)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def convert_to_features(self, batch, indices=None):
        # Tokenize text
        features = self.tokenizer.batch_encode_plus(
            batch[self.text_column],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        

        # Combine labels
        features["labels"] = [
            [float(batch[col][i]) for col in self.label_columns]
            for i in range(len(batch[self.text_column]))
        ]

        return features


if __name__ == "__main__":
    # Test the AutoTokenizerDataModule
    print("Testing AutoTokenizerDataModule...")

    # Initialize the datamodule with test parameters
    dm = AutoTokenizerDataModule(
        batch_size=64,
        max_seq_length=256,
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
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")

    print("\nTest completed successfully!")
