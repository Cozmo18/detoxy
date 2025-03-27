import os
from dataclasses import field

# from multiprocessing import cpu_count
from pathlib import Path

from pydantic import Field
from pydantic.dataclasses import dataclass

this_file = Path(__file__)
root_path = this_file.parents[2]

LABELS: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


@dataclass
class Config:
    cache_dir: str = Field(default=os.path.join(root_path, "data"))
    log_dir: str = Field(default=os.path.join(root_path, "logs"))
    ckpt_dir: str = Field(default=os.path.join(root_path, "checkpoints"))
    perf_dir: str = Field(default=os.path.join(root_path, "logs", "perf"))
    seed: int = Field(default=42, ge=0)


@dataclass
class DataModuleConfig:
    dataset_name: str = Field(default="anitamaxvim/jigsaw-toxic-comments")
    text_col: str = Field(default="comment_text")
    label_cols: list[str] = field(default_factory=lambda: LABELS)
    num_labels: int = Field(default=len(LABELS), ge=0)
    batch_size: int = Field(default=16, ge=0)
    max_length: int = Field(default=100, ge=0)
    train_split: str = Field(default="train")
    test_split: str = Field(default="test")
    train_size: float = Field(default=0.85, ge=0.0, le=1.0)
    num_workers: int = Field(default=0, ge=0)


@dataclass
class ModuleConfig:
    model_name: str = Field(default="google/bert_uncased_L-4_H-512_A-8")
    learning_rate: float = Field(default=3e-5, ge=0.0)
    finetuned: str = Field(
        default="checkpoints/google/bert_uncased_L-4_H-512_A-8_finetuned.ckpt"
    )


@dataclass
class TrainerConfig:
    accelerator: str = Field(default="auto")
    devices: int | str = Field(default="auto")
    strategy: str = Field(default="auto")
    precision: str | None = Field(default="16-mixed")
    max_epochs: int = Field(default=1, ge=1)
    deterministic: bool = Field(default=True)
    log_every_n_steps: int | None = Field(default=50, ge=1)
    num_sanity_val_steps: int | None = Field(default=1, ge=1)
