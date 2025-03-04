import pandas as pd
import tensorflow as tf

from toxy_bot.utils import config


def convert_to_tf_datasets(batch_size: int) -> None:
    PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
    TF_DATA_DIR = config.TENSORFLOW_DATA_DIR

    TF_DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for split in ["train", "val", "test"]:
        file_path = PROCESSED_DATA_DIR / f"{split}.csv"
        assert file_path.exists(), f"Missing file: {file_path}"

        df = pd.read_csv(file_path)
        datasets[split] = _df_to_tf_dataset(df, batch_size)
        datasets[split].save(str(TF_DATA_DIR / split))

    print(f"TensorFlow datasets saved to {TF_DATA_DIR}")
    for split, ds in datasets.items():
        print(f"{split.capitalize()} batches: {len(ds)}")


def _df_to_tf_dataset(df: pd.DataFrame, batch_size: int) -> tf.data.Dataset:
    features = df[config.DATASET_FEATURES].values
    labels = df[config.DATASET_LABELS].values
    return tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)


if __name__ == "__main__":
    convert_to_tf_datasets(batch_size=128)
