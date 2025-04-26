import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
from lightning.pytorch import Trainer


def get_num_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device_name() -> str:
    if torch.cuda.is_available():
        return str(torch.cuda.get_device_name().replace(" ", "-"))
    else:
        return str(torch.cpu.current_device().replace(" ", "-"))


def create_dirs(dirs: str | list[str]) -> None:
    if isinstance(dirs, str):
        dirs = [dirs]

    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)


