import json
import os
import shutil
from pathlib import Path

import torch
from lightning.pytorch import Trainer


def log_perf(
    start: float,
    stop: float,
    perf_dir: str | Path,
    trainer: Trainer,
) -> None:
    # Sync to last checkpoint
    checkpoint_callbacks = [
        callback
        for callback in trainer.callbacks
        if callback.__class__.__name__ == "ModelCheckpoint"
    ]

    if checkpoint_callbacks:
        checkpoint_callback = checkpoint_callbacks[0]
        checkpoint_path = checkpoint_callback._last_checkpoint_saved
        checkpoint_filename = checkpoint_path.split("/")[-1]
        checkpoint_version = checkpoint_filename.split(".")[0]
    else:  # this should never be triggered since the example forces use of ModelCheckpoint
        existing_perf_files = os.listdir(perf_dir)
        checkpoint_version = f"version_{len(existing_perf_files)}"

    performance_metrics = {
        "perf": {
            "device_name": torch.cuda.get_device_name(),
            "num_mode": trainer.num_nodes,
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

    performance_file_path = os.path.join(perf_dir, checkpoint_version + ".json")
    with open(performance_file_path, "w") as perf_file:
        json.dump(performance_metrics, perf_file, indent=4)


def create_dirs(dirs: str | list) -> None:
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
