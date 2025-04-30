import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from decimal import Decimal

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


def make_exp_name(
    model_name: str,
    learning_rate: float,
    batch_size: int,
    max_seq_length: int,
) -> str:
    short_model_name = model_name.replace("/", "-").replace("_", "-")
    lr = f"{Decimal(learning_rate):.0e}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    return (
        f"{short_model_name}_LR{str(lr)}_BS{batch_size}_MSL{max_seq_length}_{timestamp}"
    )


if __name__ == "__main__":
    name = make_exp_name(
        model_name="google/bert-based-uncased",
        learning_rate=5e-3,
        batch_size=32,
        max_seq_length=128,
    )
    print(name)
