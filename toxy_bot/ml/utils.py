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


def create_experiment_name(
    model_name: str,
    learning_rate: float,
    batch_size: int,
    max_length: int,
) -> str:
    model_str = model_name.replace("/", "_")
    timestamp = datetime.now().isoformat()

    elements = {
        "model": model_str,
        "device": get_device_name(),
        "lr": f"{learning_rate:.2e}",
        "bs": str(batch_size),
        "ml": str(max_length),
        "time": timestamp,
    }

    # Join with '=' between key-value pairs and '__' between different elements
    return "__".join(f"{k}={v}" for k, v in elements.items())


def parse_run_name(run_name: str) -> dict:
    """Parse a run name back into its constituent parts."""
    return dict(element.split("=") for element in run_name.split("__"))


def log_perf(
    start: float,
    stop: float,
    trainer: Trainer,
    perf_dir: str | Path,
    version: str,
) -> None:
    perf_metrics: dict[str, dict[str, str | int | float]] = {
        "perf": {
            "device_name": get_device_name(),
            "num_node": trainer.num_nodes,
            "num_devices:": trainer.num_devices,
            "strategy": trainer.strategy.__class__.__name__,
            "precision": trainer.precision,
            "epochs": trainer.current_epoch,
            "global_step": trainer.global_step,
            "max_epochs": trainer.max_epochs,
            "min_epochs": trainer.min_epochs,
            "batch_size": trainer.datamodule.batch_size,
            "num_params": f"{get_num_trainable_params(trainer.model.model):,}",
            "runtime_min": f"{(stop - start) / 60:.2f}",
        }
    }

    if not os.path.isdir(perf_dir):
        os.mkdir(perf_dir)

    perf_file = f"{perf_dir}/{version}.json"

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
