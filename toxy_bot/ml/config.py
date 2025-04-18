import os
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path

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


@dataclass(frozen=True)
class Config:
    cache_dir: str = field(default_factory=lambda: os.path.join(root_path, "data"))
    log_dir: str = field(default_factory=lambda: os.path.join(root_path, "logs"))
    ckpt_dir: str = field(
        default_factory=lambda: os.path.join(root_path, "checkpoints")
    )
    seed: int = 0


@dataclass(frozen=True)
class DataModuleConfig:
    dataset_name: str = "anitamaxvim/toxy-dataset"
    text_column: str = "text"
    label_columns: list[str] = field(default_factory=lambda: LABELS)
    num_labels: int = len(LABELS)
    train_split: str = "balanced_train"
    test_split: str = "test"
    batch_size: int = 64
    max_seq_length: int = 512
    train_size: float = 0.80
    stratify_by_column: str = "toxic"
    num_workers: int = field(default_factory=cpu_count)


@dataclass(frozen=True)
class ModuleConfig:
    model_name: str = "google-bert/bert-base-uncased"
    learning_rate: float = 2e-5
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    finetuned: str = "checkpoints/bert-base-uncased_finetuned_2025-04-11_14-07-13.ckpt"


@dataclass(frozen=True)
class TrainerConfig:
    accelerator: str = "auto"
    devices: int | str = "auto"
    strategy: str = "auto"
    precision: str | None = "16-mixed"
    max_epochs: int = 10
    log_every_n_steps: int | None = 30
    deterministic: bool = True


CONFIG = Config()
DATAMODULE_CONFIG = DataModuleConfig()
MODULE_CONFIG = ModuleConfig()
TRAINER_CONFIG = TrainerConfig()
