import os
from pathlib import Path

import lightning.pytorch as pl
import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG

class AutoTokenizerDataModule(pl.LightningDataModule):
    loader_columns = ["input_ids", "token_type_ids", "attention_mask", "labels"]
    
    def __init__(
        self,
        dataset_name: str = DATAMODULE_CONFIG.dataset_name,
        model_name: str = MODULE_CONFIG.model_name,
        text_col: str = DATAMODULE_CONFIG.text_col,
        label_cols: list[str] = DATAMODULE_CONFIG.label_cols,
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
        self.text_col = text_col
        self.label_cols = label_cols
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.train_split = train_split
        self.test_split = test_split
        self.train_size = train_size
        self.stratify_by_column = stratify_by_column
        self.num_workers = num_workers
        self.seed = seed
        
        self.num_labels = len(self.label_cols)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, use_fast=True)
        
    def prepare_data(self):
        pl.seed_everything(seed=self.seed)

        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        datasets.load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, use_fast=True)

    def setup(self, stage=None):
        # Load and split data
        self.dataset = datasets.load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        train_val_split = self.dataset[self.train_split].train_test_split(
            train_size=self.train_size, 
        )
        self.dataset = {
            "train": train_val_split["train"],
            "validation": train_val_split["test"],
            "test": self.dataset[self.test_split],
        }
        self.dataset = datasets.DatasetDict(self.dataset)
        
        # Tokenize text and combine label columns
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=self.label_cols,
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)
            
        del train_val_split
        
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.batch_size, num_workers=self.num_workers)
        
    def convert_to_features(self, batch, indices=None):
        # Tokenize text
        features = tokenize_text(
            batch[self.text_col],
            model_name=self.model_name, 
            cache_dir=self.cache_dir, 
            max_seq_length=self.max_seq_length,
        )
        
        # Combine labels
        features["labels"] = [[float(batch[col][i]) for col in self.label_cols] for i in range(len(batch[self.text_col]))]
        
        return features
        

def tokenize_text(
    text: str | list[str],
    *,
    model_name: str,
    cache_dir: str | Path,
    max_seq_length: int,
) -> dict[str, list[int | float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)

    return tokenizer.batch_encode_plus(
        text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

if __name__ == "__main__":
    # Test the AutoTokenizerDataModule
    print("Testing AutoTokenizerDataModule...")
    
    # Initialize the datamodule with test parameters
    dm = AutoTokenizerDataModule(
        batch_size=8,
        max_seq_length=128,
        train_size=0.5,
    )
    
    # Test prepare_data
    print("Testing prepare_data...")
    dm.prepare_data()
    
    # Test setup
    print("Testing setup...")
    dm.setup()
    
    # Test dataloaders
    print("Testing dataloaders...")
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()
    
    # Print some basic information
    print(f"Number of training batches: {len(train_dl)}")
    print(f"Number of validation batches: {len(val_dl)}")
    print(f"Number of test batches: {len(test_dl)}")
    
    # Test a single batch
    print("\nTesting a single batch...")
    batch = next(iter(train_dl))
    print(batch)
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    print("\nTest completed successfully!")
