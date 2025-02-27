from pathlib import Path

import pandas as pd
import tensorflow as tf

from toxy_bot.utils.config import CONFIG


def convert_to_tf_datasets() -> None:
    PROCESSED_DATA_DIR = Path(CONFIG["paths"]["processed_data"])
    TF_DATA_DIR = Path(CONFIG["paths"]["tensorflow_data"])
    BATCH_SIZE = CONFIG["dataset"]["batch_size"]

    TF_DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for split in ["train", "val", "test"]:
        file_path = PROCESSED_DATA_DIR / f"{split}.csv"
        assert file_path.exists(), f"Missing file: {file_path}"

        df = pd.read_csv(file_path)
        datasets[split] = _df_to_tf_dataset(df, BATCH_SIZE)
        datasets[split].save(str(TF_DATA_DIR / split))

    print(f"TensorFlow datasets saved to {TF_DATA_DIR}")
    for split, ds in datasets.items():
        print(f"{split.capitalize()} batches: {len(ds)}")


def _df_to_tf_dataset(df: pd.DataFrame, batch_size: int) -> tf.data.Dataset:
    features = df[CONFIG["dataset"]["features"]].values
    labels = df[CONFIG["dataset"]["labels"]].values
    return tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)


if __name__ == "__main__":
    convert_to_tf_datasets()
