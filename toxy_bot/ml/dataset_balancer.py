import os
from typing import List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from toxy_bot.ml.config import Config, DataModuleConfig


def load_and_prepare_dataset(
    dataset_name: str,
    cache_dir: str,
    label_cols: List[str],
) -> DatasetDict:
    """
    Load and prepare the dataset from Hugging Face Hub.

    Args:
        dataset_name: Name of the dataset to load
        cache_dir: Directory to cache the dataset
        label_cols: List of column names containing binary labels

    Returns:
        DatasetDict containing the loaded dataset

    Raises:
        ValueError: If the dataset doesn't contain the required label columns
    """
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    # Validate that all label columns exist in the dataset
    for split in dataset:
        missing_cols = [col for col in label_cols if col not in dataset[split].features]
        if missing_cols:
            raise ValueError(f"Missing label columns in {split} split: {missing_cols}")

    return dataset


def balance_dataset(
    df: pd.DataFrame,
    labels: List[str],
    pos_neg_ratio: float,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Balance a dataset by undersampling the negative class to achieve a specified positive-negative ratio.

    Args:
        df: Input DataFrame containing the data
        labels: List of column names containing binary labels
        pos_neg_ratio: Desired ratio of positive samples (e.g., 0.3 for 30:70 ratio)
        random_state: Random seed for reproducibility

    Returns:
        Balanced DataFrame with the specified positive-negative ratio

    Raises:
        ValueError: If pos_neg_ratio is not between 0 and 1
    """
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


def save_dataset_splits(
    dataset_dict: DatasetDict,
    cache_dir: str,
    splits: List[str] = ["train", "test", "balanced_train"],
) -> None:
    """
    Save dataset splits to parquet files.

    Args:
        dataset_dict: DatasetDict containing the splits to save
        cache_dir: Directory to save the parquet files
        splits: List of split names to save

    Raises:
        ValueError: If a requested split doesn't exist in the dataset
    """
    for split in splits:
        if split not in dataset_dict:
            raise ValueError(f"Split {split} not found in dataset")

        output_path = os.path.join(cache_dir, f"{split}.parquet")
        dataset_dict[split].to_parquet(output_path)


def create_balanced_dataset(
    config: Config,
    dm_config: DataModuleConfig,
    pos_neg_ratio: float = 0.3,
) -> DatasetDict:
    """
    Create a balanced version of the dataset.

    Args:
        config: Main configuration object
        dm_config: Data module configuration object
        pos_neg_ratio: Desired ratio of positive samples

    Returns:
        DatasetDict containing the original and balanced splits
    """
    # Load and validate the dataset
    dataset = load_and_prepare_dataset(
        dataset_name=dm_config.dataset_name,
        cache_dir=config.cache_dir,
        label_cols=dm_config.label_cols,
    )

    # Balance the train dataset
    train_df = dataset["train"].to_pandas()
    balanced_train_df = balance_dataset(
        df=train_df,
        labels=dm_config.label_cols,
        pos_neg_ratio=pos_neg_ratio,
        random_state=config.seed,
    )

    # Create balanced dataset
    balanced_train_dataset = Dataset.from_pandas(balanced_train_df)
    new_dataset_dict = DatasetDict(
        {
            "train": dataset["train"],
            "test": dataset["test"],
            "balanced_train": balanced_train_dataset,
        }
    )

    # Save the dataset splits
    save_dataset_splits(new_dataset_dict, config.cache_dir)

    return new_dataset_dict


if __name__ == "__main__":
    config = Config()
    dm_config = DataModuleConfig()

    try:
        new_dataset_dict = create_balanced_dataset(config, dm_config)

        print("\nBalanced dataset statistics:")
        for split in new_dataset_dict:
            n_samples = len(new_dataset_dict[split])
            print(f"{split}: {n_samples} samples")

    except Exception as e:
        print(f"Error creating balanced dataset: {str(e)}")
        raise
