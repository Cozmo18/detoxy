import shutil
from enum import Enum
from pathlib import Path

import kagglehub

from toxy_bot.utils.config import CONFIG


class Files(Enum):
    TRAIN = "train.csv"
    TEST = "test.csv"
    TEST_LABELS = "test_labels.csv"


def download_dataset(force_download: bool = False) -> None:
    RAW_DATA_DIR = Path(CONFIG["paths"]["raw_data"])
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    handle = CONFIG["dataset"]["kaggle_handle"]
    download_dir = Path(
        kagglehub.dataset_download(
            handle=handle,
            force_download=force_download,
        )
    )

    for file in Files:
        source = download_dir / file.value
        destination = RAW_DATA_DIR / file.value
        assert source.exists(), f"Download failed: {file.value} is missing."
        shutil.copy2(source, destination)

    print(f"Dataset downloaded successfully to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    download_dataset()
