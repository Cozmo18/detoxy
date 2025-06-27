import os
from datetime import UTC, datetime
from pathlib import Path

import lightning.pytorch as pl
import torch
from datasets import load_dataset
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from model.config import Config, DataModuleConfig


class DatasetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = DataModuleConfig.dataset_name,
        data_dir: str | Path = Config.data_dir,
        cache_dir: str | Path = Config.cache_dir,
        text_col: str = DataModuleConfig.text_col,
        label_cols: tuple[str, ...] = DataModuleConfig.label_cols,
        train_split: str = DataModuleConfig.train_split,
        test_split: str = DataModuleConfig.test_split,
        stratify_by_col: str = DataModuleConfig.stratify_by_col,
        val_size: float = DataModuleConfig.val_size,
        batch_size: int = DataModuleConfig.batch_size,
        num_workers: int = DataModuleConfig.num_workers,
        seed: int = Config.seed,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.text_col = text_col
        self.label_cols = label_cols
        self.train_split = train_split
        self.test_split = test_split
        self.stratify_by_col = stratify_by_col
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.cols_to_keep = [*self.label_cols, self.text_col]

    def prepare_data(self) -> None:
        pl.seed_everything(seed=self.seed)

        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        cache_dir_path = Path(self.cache_dir)
        if not cache_dir_path.is_dir():
            cache_dir_path.mkdir(parents=True, exist_ok=True)

        cache_dir_is_empty = not any(cache_dir_path.iterdir())
        if cache_dir_is_empty:
            rank_zero_info(f"[{datetime.now(UTC)!s}] Downloading dataset")
            load_dataset(
                self.dataset_name,
                data_dir=self.data_dir,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        else:
            rank_zero_info(
                f"[{datetime.now(UTC)!s}] Data cache exists. Loading from cache."
            )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            dataset = load_dataset(
                self.dataset_name,
                split=self.train_split,
                data_dir=self.data_dir,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            dataset = dataset.train_test_split(  # type: ignore[attr-defined]
                test_size=self.val_size,
                stratify_by_column=self.stratify_by_col,
            )
            self.train_data = dataset["train"]
            self.train_data.set_format(type="torch", columns=self.cols_to_keep)
            self.val_data = dataset["test"]
            self.val_data.set_format(type="torch", columns=self.cols_to_keep)
            del dataset

        if stage == "test" or stage is None:
            self.test_data = load_dataset(
                self.dataset_name, split=self.test_split, cache_dir=self.cache_dir
            )
            self.test_data.set_format(type="torch", columns=self.cols_to_keep)

    def collate_fn(self, batch: list[dict]) -> tuple[list[str], torch.Tensor]:
        # batch is a list of dicts, each with label columns and text_col
        texts = [item[self.text_col] for item in batch]
        # Stack label columns into a single tensor (batch_size, num_labels)
        labels = torch.stack(
            [
                torch.tensor(
                    [item[label] for label in self.label_cols], dtype=torch.float
                )
                for item in batch
            ]
        )
        return texts, labels

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":
    dm = DatasetDataModule()
    dm.prepare_data()
    dm.setup("fit")
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    print(f"train batches: {len(train_dl)}")
    print(f"val batches: {len(val_dl)}")
    batch = next(iter(train_dl))
    print(batch)
