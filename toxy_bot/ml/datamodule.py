import os
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
from datasets import load_dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG, MODULE_CONFIG

class AutoTokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = DATAMODULE_CONFIG.dataset_name,
        model_name: str = MODULE_CONFIG.model_name,
        cache_dir: str = CONFIG.cache_dir,
        text_col: str = DATAMODULE_CONFIG.text_col,
        label_cols: list[str] = DATAMODULE_CONFIG.label_cols,
        columns: list[str] = ["input_ids", "attention_mask", "labels"],
        batch_size: int = DATAMODULE_CONFIG.batch_size,
        max_token_len: int = DATAMODULE_CONFIG.max_token_len,
        train_split: str = DATAMODULE_CONFIG.train_split,
        test_split: str = DATAMODULE_CONFIG.test_split,
        train_size: float = DATAMODULE_CONFIG.train_size,
        num_workers: int = DATAMODULE_CONFIG.num_workers,
        seed: int = CONFIG.seed,
    ) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.text_col = text_col
        self.label_cols = label_cols
        self.columns = columns
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.train_split = train_split
        self.test_split = test_split
        self.train_size = train_size
        self.num_workers = num_workers
        self.seed = seed
        
        self.train_data = None
        self.val_data = None
        self.test_data = None

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
        if stage == "fit":
            # Load and split training data
            dataset = load_dataset(
                self.dataset_name, split=self.train_split, cache_dir=self.cache_dir
            )
            dataset = dataset.train_test_split(train_size=self.train_size)  # type: ignore

            # First, process the labels
            self.train_data = dataset["train"].map(
                combine_labels,
                batched=True,
                remove_columns=self.label_cols,
                fn_kwargs={"label_cols": self.label_cols},
            )

            # Then tokenize the text
            self.train_data = self.train_data.map(
                tokenize_text,
                batched=True,
                fn_kwargs={
                    "model_name": self.model_name,
                    "cache_dir": self.cache_dir,
                    "max_token_len": self.max_token_len,
                    "text_col": self.text_col,
                },
            )
            # And convert to torch tensors
            self.train_data.set_format("torch", columns=self.columns)

            # And similarly for validation data:
            self.val_data = dataset["test"].map(
                combine_labels,
                batched=True,
                remove_columns=self.label_cols,
                fn_kwargs={"label_cols": self.label_cols},
            )

            self.val_data = self.val_data.map(
                tokenize_text,
                batched=True,
                fn_kwargs={
                    "model_name": self.model_name,
                    "cache_dir": self.cache_dir,
                    "max_token_len": self.max_token_len,
                    "text_col": self.text_col,
                },
            )
            self.val_data.set_format("torch", columns=self.columns)

            del dataset  # Free memory
            
        if stage == "test":
            self.test_data = load_dataset(
                self.dataset_name, split=self.test_split, cache_dir=self.cache_dir
            )
            
            self.test_data = self.test_data.map(
                combine_labels,
                batched=True,
                remove_columns=self.label_cols,
                fn_kwargs={"label_cols": self.label_cols},
            )

            self.test_data = self.test_data.map(
                tokenize_text,
                batched=True,
                fn_kwargs={
                    "model_name": self.model_name,
                    "cache_dir": self.cache_dir,
                    "max_token_len": self.max_token_len,
                    "text_col": self.text_col,
                },
            )
            self.test_data.set_format("torch", columns=self.columns)


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


def tokenize_text(
    batch: str | dict,
    *,
    model_name: str,
    cache_dir: str | Path,
    max_token_len: int,
    text_col: str | None = None,
    truncation: bool = True,
    add_special_tokens: bool = True,
    return_attention_mask: bool = True,
    return_token_type_ids: bool = False,
    padding: bool | str = "max_length",
) -> dict[str, list[int | float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    if isinstance(batch, str):
        text = batch
    else:
        if text_col is None:
            raise ValueError(
                "When batch is provided as a dictionary, text_col must be specified"
            )
        text = batch[text_col]

    return tokenizer(
        text,
        truncation=truncation,
        add_special_tokens=add_special_tokens,
        return_attention_mask=return_attention_mask,
        return_token_type_ids=return_token_type_ids,
        padding=padding,
        max_length=max_token_len,
        return_tensors="pt",
    )
    

def combine_labels(
    batch: dict,
    *,
    label_cols: list[str],
) -> dict:
    """Process label columns and combine them into a single labels field"""
    # Create a combined labels list for each sample
    labels = []
    for i in range(len(batch[label_cols[0]])):
        sample_labels = [float(batch[label_col][i]) for label_col in label_cols]
        labels.append(sample_labels)

    batch["labels"] = labels
    return batch


if __name__ == "__main__":
    # Test the AutoTokenizerDataModule
    print("Testing AutoTokenizerDataModule...")
    
    # Initialize the datamodule with test parameters
    test_datamodule = AutoTokenizerDataModule(
        batch_size=8,
        max_token_len=128,
        train_size=0.8,
        num_workers=0  # Set to 0 for testing to avoid multiprocessing issues
    )
    
    # Test prepare_data
    print("Testing prepare_data...")
    test_datamodule.prepare_data()
    
    # Test setup
    print("Testing setup...")
    test_datamodule.setup(stage="fit")
    
    # Test dataloaders
    print("Testing dataloaders...")
    train_loader = test_datamodule.train_dataloader()
    val_loader = test_datamodule.val_dataloader()
    
    # Print some basic information
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Test a single batch
    print("\nTesting a single batch...")
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    print("\nTest completed successfully!")
