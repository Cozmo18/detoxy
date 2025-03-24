import os
import shutil


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
