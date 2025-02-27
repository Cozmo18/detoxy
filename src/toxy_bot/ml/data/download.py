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
    raw_data_dir = Path(CONFIG["paths"]["raw_data"])
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle downloads to cache dir
    handle = CONFIG["dataset"]["kaggle_handle"]
    download_dir = Path(
        kagglehub.dataset_download(
            handle=handle,
            force_download=force_download,
        )
    )

    for file in Files:
        source = download_dir / file.value
        destination = raw_data_dir / file.value

        if not source.exists():
            raise FileNotFoundError(
                f"Download failed. The {file.value} file is missing."
            )
        else:
            shutil.copy2(source, destination)

    print(f"Data downloaded to {raw_data_dir}")

    return None


if __name__ == "__main__":
    download_dataset()
