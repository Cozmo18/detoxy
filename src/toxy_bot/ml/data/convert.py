from pathlib import Path

import pandas as pd
import tensorflow as tf

from toxy_bot.utils.config import CONFIG


def convert_to_tf_datasets() -> None:
    processed_data_dir = Path(CONFIG["paths"]["processed_data"])
    tf_data_dir = Path(CONFIG["paths"]["tensorflow_data"])

    train_df = pd.read_csv(processed_data_dir / "train.csv")
    val_df = pd.read_csv(processed_data_dir / "val.csv")
    test_df = pd.read_csv(processed_data_dir / "test.csv")

    train_ds = _df_to_tf_dataset(train_df)
    val_ds = _df_to_tf_dataset(val_df)
    test_ds = _df_to_tf_dataset(test_df)

    train_ds.save(str(tf_data_dir / "train"))
    val_ds.save(str(tf_data_dir / "val"))
    test_ds.save(str(tf_data_dir / "test"))

    print(f"TensorFLow datasets saved to {tf_data_dir}")
    print("Train batches: ", len(train_ds))
    print("Val batches: ", len(val_ds))
    print("Test batches: ", len(test_ds))


def _df_to_tf_dataset(df: pd.DataFrame) -> tf.data.Dataset:
    features = df[CONFIG["dataset"]["features"]].values
    labels = df[CONFIG["dataset"]["labels"]].values
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.batch(CONFIG["dataset"]["batch_size"])


if __name__ == "__main__":
    convert_to_tf_datasets()
