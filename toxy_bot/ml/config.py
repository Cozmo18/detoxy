import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

this_file = Path(__file__)
root_path = this_file.parents[2]


@dataclass
class Config:
    cache_dir: str = os.path.join(root_path, "data")
    log_dir: str = os.path.join(root_path, "logs")
    ckpt_dir: str = os.path.join(root_path, "checkpoint")
    perf_dir: str = os.path.join(root_path, "logs", "perf")
    seed: int = 0


@dataclass
class DatasetConfig:
    dataset_name: str = "julian3833/jigsaw-toxic-comment-classification-challenge"
    dataset_paths: dict[str, str] = field(
        default_factory=lambda: {
            "train": "train.csv",
            "test_inputs": "test.csv",
            "test_labels": "test_labels.csv",
        }
    )
    input_list: list[str] = field(default_factory=lambda: ["comment_text"])
    label_list: list[str] = field(
        default_factory=lambda: [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
    )


@dataclass
class DataModuleConfig:
    num_labels: int = 6
    train_size: float = 0.85
    batch_size: int = 128
    max_len: int = 256
    train_split: str = "train"
    valid_split: str = "valid"
    test_split: str = "test"


@dataclass
class ModuleConfig:
    model_name: str = "google/bert_uncased_L-4_H-512_A-8"
    leaning_rate: float = 2e-5
    finetuned: str = "checkpoints/google/bert_uncased_L-4_H-512_A-8_finetuned.ckpt"


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    devices: int | str = "auto"
    strategy: str = "auto"
    precision: Optional[str] = "16-mixed"
    max_epochs: int = 1


if __name__ == "__main__":
    pass
