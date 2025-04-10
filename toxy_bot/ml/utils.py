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


def log_perf(
    start: float,
    stop: float,
    trainer: Trainer,
    perf_dir: str | Path,
) -> None:
    # sync to last checkpoint
    mc = [i for i in trainer.callbacks if i.__class__.__name__ == "ModelCheckpoint"]
    if mc:
        mc = mc[0]
        version = mc._last_checkpoint_saved.split("/")[-1].split(".")[0]
    else: # this should never be triggered since the example forces use of ModelCheckpoint
        perfs = os.listdir(perf_dir)
        version = f"version_{len(perfs)}"
        
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
        

    with open(os.path.join(perf_dir, version + ".json"), "w") as perf_file:
        json.dump(perf_metrics, perf_file, indent=4)



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
