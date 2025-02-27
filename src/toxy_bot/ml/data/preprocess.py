from pathlib import Path
from typing import Tuple

import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

from toxy_bot.ml.data.schemas import CommentData
from toxy_bot.utils.config import CONFIG


def process_data() -> None:
    train = _load_and_validate_train()
    test = _load_and_validate_test()

    features = CONFIG["dataset"]["features"]
    labels = CONFIG["dataset"]["labels"]

    cols_to_keep = features + labels
    train = train[cols_to_keep].dropna()
    test = test[cols_to_keep].dropna()

    train = _drop_untested_samples(train, labels)
    test = _drop_untested_samples(test, labels)

    val_size = CONFIG["dataset"]["val_size"]
    train, val = _iter_split_df(train, features, labels, val_size)

    processed_data_dir = Path(CONFIG["paths"]["processed_data"])
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_data_dir / "train.csv", index=False)
    val.to_csv(processed_data_dir / "val.csv", index=False)
    test.to_csv(processed_data_dir / "test.csv", index=False)

    print(f"Processed data saved to {processed_data_dir}")
    print("Train: ", train.shape)
    print("Val: ", val.shape)
    print("Test: ", test.shape)


def _load_and_validate_train() -> pd.DataFrame:
    filepath = Path(CONFIG["paths"]["raw_data"]) / "train.csv"
    df = pd.read_csv(filepath)
    _validate_data(df)
    return df


def _load_and_validate_test() -> pd.DataFrame:
    inputs_path = Path(CONFIG["paths"]["raw_data"]) / "test.csv"
    labels_path = Path(CONFIG["paths"]["raw_data"]) / "test_labels.csv"

    inputs_df = pd.read_csv(inputs_path)
    labels_df = pd.read_csv(labels_path)

    merged_df = inputs_df.merge(labels_df, on="id", validate="one_to_one")  # type: ignore
    _validate_data(merged_df)
    return merged_df


def _validate_data(df: pd.DataFrame) -> None:
    samples = df.to_dict(orient="records")
    valid_samples = [CommentData(**row) for row in samples]
    return None


def _iter_split_df(
    df: pd.DataFrame,
    features: list[str],
    labels: list[str],
    test_size: float,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1, random_state=0) if shuffle else df

    X = df[features].values.reshape(-1, 1)  # type: ignore
    y = df[labels].values  # type: ignore

    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X.reshape(-1, 1), y, test_size
    )

    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)

    train_df[features] = X_train
    train_df[labels] = y_train
    test_df[features] = X_test
    test_df[labels] = y_test

    assert len(train_df) + len(test_df) == len(df), "Size mismatch"

    return train_df, test_df


def _drop_untested_samples(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    binary_condition = df[labels].isin([0, 1]).all(axis=1)  # type: ignore
    return df[binary_condition].reset_index(drop=True)  # type: ignore


if __name__ == "__main__":
    process_data()
