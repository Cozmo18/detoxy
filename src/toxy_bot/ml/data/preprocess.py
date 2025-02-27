from pathlib import Path
from typing import Tuple

import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

from toxy_bot.ml.data.schemas import CommentData
from toxy_bot.utils.config import CONFIG


def process_data() -> None:
    train = _load_data("train.csv")
    test = _load_data("test.csv", "test_labels.csv")

    features = CONFIG["dataset"]["features"]
    labels = CONFIG["dataset"]["labels"]

    train, test = [df[features + labels].dropna() for df in (train, test)]
    train, test = map(lambda df: _drop_untested_samples(df, labels), (train, test))
    train, val = _iter_split_df(train, features, labels, CONFIG["dataset"]["val_size"])

    _save_data(train, val, test)

    print(f"Processed data saved to {CONFIG['paths']['processed_data']}")
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")


def _load_data(inputs_file: str, labels_file: str | None = None) -> pd.DataFrame:
    RAW_DATA_DIR = Path(CONFIG["paths"]["raw_data"])
    inputs_df = pd.read_csv(RAW_DATA_DIR / inputs_file)

    if labels_file:
        labels_df = pd.read_csv(RAW_DATA_DIR / labels_file)
        df = inputs_df.merge(labels_df, on="id", validate="one_to_one")
    else:
        df = inputs_df

    _validate_data(df)
    return df


def _validate_data(df: pd.DataFrame) -> None:
    for row in df.to_dict(orient="records"):
        CommentData(**row)  # type: ignore


def _iter_split_df(
    df: pd.DataFrame, features: list[str], labels: list[str], test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1, random_state=0)

    X, y = df[features].values, df[labels].values

    # X = df[features].values.reshape(-1, 1)  # type: ignore
    # y = df[labels].values  # type: ignore

    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size)

    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)

    train_df[features], train_df[labels] = X_train, y_train
    test_df[features], test_df[labels] = X_test, y_test

    return train_df, test_df


def _drop_untested_samples(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    return df.loc[df[labels].isin([0, 1]).all(axis=1)].reset_index(drop=True)  # type: ignore


def _save_data(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR = Path(CONFIG["paths"]["processed_data"])
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split, df in zip(["train", "val", "test"], [train, val, test]):
        df.to_csv(PROCESSED_DATA_DIR / f"{split}.csv", index=False)


if __name__ == "__main__":
    process_data()
