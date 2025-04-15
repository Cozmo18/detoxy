import os
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path

this_file = Path(__file__)
root_path = this_file.parents[2]

@dataclass(frozen=True)
class Config:
    cache_dir: str = field(default_factory=lambda: os.path.join(root_path, "data"))
    log_dir: str = field(default_factory=lambda: os.path.join(root_path, "logs"))
    ckpt_dir: str = field(default_factory=lambda: os.path.join(root_path, "checkpoints"))
    perf_dir: str = field(default_factory=lambda: os.path.join(root_path, "logs", "perf"))
    seed: int = 42


@dataclass(frozen=True)
class DataModuleConfig:
    dataset_name: str = "anitamaxvim/jigsaw-toxic-comments"
    text_col: str = "comment_text"
    label_cols: list[str] = field(default_factory=lambda: ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    num_labels: int = 6 
    train_split: str = "balanced_train"
    test_split: str = "test"
    batch_size: int = 128
    max_seq_length: int = 512
    train_size: float = 0.80
    stratify_by_column: str = "toxic"
    num_workers: int = field(default_factory=cpu_count)


@dataclass(frozen=True)
class ModuleConfig:
    model_name: str = "google/bert_uncased_L-2_H-128_A-2" #"google-bert/bert-base-uncased"
    learning_rate: float = 2e-5
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    weight_decay: float = 0
    # finetuned: str = "checkpoints/google/bert_uncased_L-4_H-512_A-8_finetuned.ckpt"


@dataclass(frozen=True)
class TrainerConfig:
    accelerator: str = "auto"
    devices: int | str = "auto"
    strategy: str = "auto"
    precision: str | None = "16-mixed"
    max_epochs: int = 5
    deterministic: bool = True
    check_val_every_n_epoch: int | None = 1
    val_check_interval: int | float | None = 0.5 # x2 per training epoch
    num_sanity_val_steps: int | None = 2
    log_every_n_steps: int | None = 50
    
CONFIG = Config()
DATAMODULE_CONFIG = DataModuleConfig()
MODULE_CONFIG = ModuleConfig()
TRAINER_CONFIG = TrainerConfig()

