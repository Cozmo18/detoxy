from datasets import load_dataset, Dataset, ClassLabel
import pandas as pd
import os
from toxy_bot.ml.config import CONFIG, DATAMODULE_CONFIG


def balance_dataset(
    df: pd.DataFrame,
    labels: list[str],
    pos_neg_ratio: float,
    random_state: int = CONFIG.seed,
) -> pd.DataFrame:
    if not 0 < pos_neg_ratio < 1:
        raise ValueError("pos_neg_ratio must be between 0 and 1")

    # Split into positive and negative samples
    neg_df = df[df[labels].sum(axis=1) == 0]
    pos_df = df[df[labels].sum(axis=1) > 0]

    # Calculate how many negative samples we need based on the ratio
    n_pos = len(pos_df)
    n_neg_needed = int(n_pos * (1 - pos_neg_ratio) / pos_neg_ratio)

    # Randomly sample negative samples
    neg_df_sampled = neg_df.sample(n=n_neg_needed, random_state=random_state)

    # Combine positive and sampled negative samples
    balanced_df = pd.concat([pos_df, neg_df_sampled], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(
        drop=True
    )

    return balanced_df


if __name__ == "__main__":
    train_ds = load_dataset(
        "anitamaxvim/jigsaw-toxic-comments", split="train", cache_dir=CONFIG.cache_dir
    )
    test_ds = load_dataset(
        "anitamaxvim/jigsaw-toxic-comments", split="test", cache_dir=CONFIG.cache_dir
    )

    label_columns = DATAMODULE_CONFIG.labels

    train_df = train_ds.to_pandas()
    balanced_train_df = balance_dataset(train_df, label_columns, pos_neg_ratio=0.25)
    balanced_train_ds = Dataset.from_pandas(balanced_train_df)

    for label_col in label_columns:
        balanced_train_ds = balanced_train_ds.cast_column(
            label_col, ClassLabel(names=["0", "1"])
        )

        train_ds = train_ds.cast_column(
            label_col,
            ClassLabel(names=["0", "1"]),
        )

        test_ds = test_ds.cast_column(label_col, ClassLabel(names=["0", "1"]))

    balanced_train_ds = balanced_train_ds.rename_column("comment_text", "text")
    train_ds = train_ds.rename_column("comment_text", "text")
    test_ds = test_ds.rename_column("comment_text", "text")

    balanced_train_ds.to_parquet(
        os.path.join(CONFIG.cache_dir, "balanced_train.parquet")
    )
    train_ds.to_parquet(os.path.join(CONFIG.cache_dir, "full_train.parquet"))
    test_ds.to_parquet(os.path.join(CONFIG.cache_dir, "test.parquet"))
