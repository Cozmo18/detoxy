import os
from dataclasses import dataclass
from pathlib import Path

this_file = Path(__file__)
root_path = this_file.parents[2]


@dataclass(frozen=True)
class Config:
    predict_url: str = os.getenv("ML_SERVICE_URL", "http://0.0.0.0:8000/predict")
    threshold: float = float(os.getenv("TOXICITY_THRESHOLD", "0.8"))
    log_dir: str | Path = os.path.join(root_path, "logs", "discord.log")
