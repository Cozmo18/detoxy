import os
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN: str | None = os.environ.get("HF_TOKEN")
COMET_API_KEY: str | None = os.environ.get("COMET_API_KEY")
COMET_WORKSPACE: str | None = os.environ.get("COMET_WORKSPACE")


THIS_FILE = Path(__file__)
ROOT_PATH = THIS_FILE.parents[2]


@dataclass()
class Config:
    data_dir: str = str(ROOT_PATH / "data")
    cache_dir: str = str(ROOT_PATH / "data")
    log_dir: str = str(ROOT_PATH / "logs")
    ckpt_dir: str = str(ROOT_PATH / "checkpoints")
    seed: int = 0


@dataclass
class DataModuleConfig:
    dataset_name: str = "google/jigsaw_toxicity_pred"
    text_col: str = "comment_text"
    label_cols: tuple[str, ...] = ("toxic", "threat")
    train_split: str = "train"
    test_split: str = "test"
    stratify_by_col: str = "toxic"
    val_size: float = 0.2
    batch_size: int = 64
    stratify_by_column: str = "toxic"
    num_workers: int = cpu_count()


@dataclass
class ModuleConfig:
    model_name: str = "prajjwal1/bert-tiny"
    max_token_len: int = 128
    learning_rate: float = 3e-5
    finetuned: Path = field(
        default_factory=lambda: THIS_FILE.parent
        / "checkpoints"
        / "google-bert-uncased-L-2-H-128-A-2_LR3e-5_BS64_MSL512_20250430-161306.ckpt"
    )


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    devices: int | str = "auto"
    strategy: str = "auto"
    precision: str | None = "16-mixed"
    max_epochs: int = 5


@dataclass
class AppConfig:
    finetuned: str = (
        "google-bert-uncased-L-2-H-128-A-2_LR3e-5_BS64_MSL512_20250430-161306.ckpt"
    )
    accelerator: str = "auto"
    devices: int | str = "auto"
    timeout: int = 30
    track_requests: bool = True
    generate_client_file: bool = False


CONIFG = Config()
DM_CONFIG = DataModuleConfig()
