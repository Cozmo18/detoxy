import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from torchmetrics.functional import accuracy

this_file = Path(__file__)
root_path = this_file.parents[2]

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


@dataclass
class Config:
    cache_dir: str = os.path.join(root_path, "data", "huggingface")
    external_dir: Optional[str] = os.path.join(root_path, "data", "external")
    log_dir: str = os.path.join(root_path, "logs")
    ckpt_dir: str = os.path.join(root_path, "checkpoint")
    perf_dir: str = os.path.join(root_path, "logs", "perf")
    seed: int = 42


@dataclass
class DataModuleConfig:
    dataset_name: str = "google/jigsaw_toxicity_pred"
    text_col: str = "comment_text"
    label_cols: list[str] = field(default_factory=lambda: LABELS)
    num_labels: int = len(LABELS)
    batch_size: int = 12
    max_length: int = 100
    train_split: str = "train"
    test_split: str = "test"
    train_size: float = 0.85
    num_workers: int = 0  # cpu_count()


@dataclass
class ModuleConfig:
    model_name: str = "textattack/bert-base-uncased-yelp-polarity"  # "google/bert_uncased_L-4_H-512_A-8"
    learning_rate: float = 5e-5
    accuracy: Any = accuracy  # TODO: fix type
    finetuned: str = "checkpoints/google/bert_uncased_L-4_H-512_A-8_finetuned.ckpt"


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    devices: int | str = "auto"
    strategy: str = "auto"
    precision: Optional[str] = "bf16-mixed"  # "16-mixed"
    max_epochs: int = 1


if __name__ == "__main__":
    pass
