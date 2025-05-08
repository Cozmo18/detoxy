import os
from dataclasses import dataclass, field
from pathlib import Path

this_file = Path(__file__)
root_path = this_file.parents[2]


@dataclass(frozen=True)
class Config:
    log_dir: str | Path = field(default_factory=lambda: os.path.join(root_path, "logs", "discord.log"))