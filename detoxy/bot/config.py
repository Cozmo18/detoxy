import os
from dataclasses import dataclass
from pathlib import Path

this_file = Path(__file__)
root_path = this_file.parents[2]


@dataclass(frozen=True)
class Config:
    threshold: float = 0.8
    log_dir: str | Path = os.path.join(root_path, "logs", "discord.log")
