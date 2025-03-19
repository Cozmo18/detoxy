import os

import kagglehub
import pandas as pd
from datasets import Dataset, DatasetDict
from kagglehub import KaggleDatasetAdapter

from toxy_bot.ml.config import Config, DatasetConfig
from toxy_bot.ml.utils import create_dirs


def _download_dataset_from_kaggle(
    dataset_name: str,
    dataset_paths: dict[str, str],
    train_split: str,
    test_split: str,
) -> dict[str, pd.DataFrame]:
    """Download dataset files from Kaggle and load them into pandas DataFrames."""
    train_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset_name,
        dataset_paths["train"],
    )

    test_inputs_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset_name,
        dataset_paths["test_inputs"],
    )

    test_labels_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset_name,
        dataset_paths["test_labels"],
    )

    test_df = pd.merge(
        test_inputs_df,
        test_labels_df,
        on="id",
        how="inner",
        validate="1:1",
    )

    df_dict = {}
    df_dict[train_split] = train_df
    df_dict[test_split] = test_df

    return df_dict


def _clean_and_prepare_dataset(
    df_dict: dict[str, pd.DataFrame],
    label_list: list[str],
    train_split: str,
    test_split: str,
) -> DatasetDict:
    """Process and clean the dataset, returning a HuggingFace DatasetDict."""
    dataset = DatasetDict()
    for split, df in df_dict.items():
        # Clean up dataframe by removing invalid samples
        # (those with -1 in any label column)
        invalid_samples = df.loc[(df[label_list].values == -1)].index
        df = df.drop(index=invalid_samples).reset_index(drop=True)
        df = df.drop(columns="id")  # ID not needed for ML processing

        # Convert to HuggingFace Dataset
        ds = Dataset.from_pandas(df)
        dataset[split] = ds

    return dataset


def load_dataset(
    dataset_name: str = DatasetConfig.dataset_name,
    dataset_paths: dict[str, str] = DatasetConfig().dataset_paths,
    label_list: list[str] = DatasetConfig().label_list,
    train_size: float = DatasetConfig.train_size,
    train_split: str = DatasetConfig.train_split,
    valid_split: str = DatasetConfig.valid_split,
    test_split: str = DatasetConfig.test_split,
    cache_dir: str = Config.cache_dir,
    force_download: bool = False,
) -> DatasetDict:
    """
    Load dataset from cache if available, otherwise download from Kaggle,
    process it, and save to cache.

    Args:
        config: Dataset configuration
        module_config: Data module configuration for split names
        cache_dir: Directory to store cached datasets
        force_download: If True, always download fresh data even if cache exists

    Returns:
        HuggingFace DatasetDict with train, valid, and test splits
    """
    # Create dataset directory path
    dataset_dir = os.path.join(cache_dir, dataset_name)

    # Check if the dataset is already cached
    if os.path.exists(dataset_dir) and not force_download:
        print(f"Loading dataset from cache: {dataset_dir}")
        dataset = DatasetDict.load_from_disk(dataset_dir)
    else:
        # Download fresh data if no cache or force_download is True
        print(f"Downloading dataset from Kaggle: {dataset_name}")
        create_dirs(dataset_dir)

        # Load raw kaggle dataset into dictionary of Pandas dataframes
        df_dict = _download_dataset_from_kaggle(
            dataset_name, dataset_paths, train_split, test_split
        )

        # Clean up dataframes and convert to HuggingFace DatasetDict
        dataset = _clean_and_prepare_dataset(
            df_dict, label_list, train_split, test_split
        )

        # Save processed dataset to cache for future use
        print(f"Saving dataset to cache: {dataset_dir}")
        dataset.save_to_disk(dataset_dir)

    # Create train/validation
    print(f"Creating train/validation split with {train_size:.0%} training data")
    train_valid_ds = dataset[train_split].train_test_split(train_size=train_size)
    dataset[train_split] = train_valid_ds["train"]
    dataset[valid_split] = train_valid_ds["test"]

    return dataset


if __name__ == "__main__":
    datasets = load_dataset()
    for split_name, dataset in datasets.items():
        print(f"{split_name}: {len(dataset)} examples")
    print(f"Example features: {list(datasets['train'].features.keys())}")
