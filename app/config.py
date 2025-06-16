import os
from dataclasses import dataclass
from pathlib import Path

this_file = Path(__file__)
root_path = this_file.parent


@dataclass(frozen=True)
class Config:
    threshold: float = 0.8
