import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
from lightning.pytorch import Trainer


def create_run_name(
    model_name: str,
    learning_rate: float,
    batch_size: int,
    max_length: int,
) -> str:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
    else:
        device_name = "CPU"

    model_str = model_name.replace("/", "_")
    timestamp = datetime.now().isoformat()

    return f"{model_str}_{device_name}_LR{learning_rate}_BS{batch_size}_ML{max_length}_{timestamp}"


def log_perf(
    start: float,
    stop: float,
    perf_dir: str | Path,
    trainer: Trainer,
    run_name: str,
) -> None:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
    else:
        device_name = "CPU"

    perf_metrics = {
        "perf": {
            "device_name": device_name,
            "num_node": trainer.num_nodes,
            "num_devices:": trainer.num_devices,
            "strategy": trainer.strategy.__class__.__name__,
            "precision": trainer.precision,
            "epochs": trainer.current_epoch,
            "global_step": trainer.global_step,
            "max_epochs": trainer.max_epochs,
            "min_epochs": trainer.min_epochs,
            "batch_size": trainer.datamodule.batch_size,
            "runtime_min": (stop - start) / 60,
        }
    }

    if not os.path.isdir(perf_dir):
        os.mkdir(perf_dir)

    perf_file = f"{perf_dir}/{run_name}.json"

    with open(perf_file, "w") as f:
        json.dump(perf_metrics, f, indent=4)


def create_dirs(dirs: str | list[str]) -> None:
    if isinstance(dirs, str):
        dirs = [dirs]

    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)


def copy_dir_contents(source_dir: str, target_dir: str) -> None:
    """
    Copy all contents from source directory to target directory.
    Creates target directory if it doesn't exist.

    Args:
        source_dir: Path to the source directory
        target_dir: Path to the target directory
    """
    # Check if source directory exists
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist")

    # Create target directory if it doesn't exist
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    # Copy all files and subdirectories
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        target_item = os.path.join(target_dir, item)

        if os.path.isdir(source_item):
            # If it's a directory, copy the entire directory
            shutil.copytree(source_item, target_item, dirs_exist_ok=True)
        else:
            # If it's a file, copy the file
            shutil.copy2(source_item, target_item)
