import os

import kagglehub
import pandas as pd
from datasets import Dataset, DatasetDict
from kagglehub import KaggleDatasetAdapter

from toxy_bot.ml.config import Config, DataModuleConfig, DatasetConfig
from toxy_bot.ml.utils import create_dirs


def load_dataset(
    dataset_name: str = DatasetConfig.dataset_name,
    dataset_paths: dict[str, str] = DatasetConfig().dataset_paths,
    label_list: list[str] = DatasetConfig().label_list,
    train_split: str = DataModuleConfig.train_split,
    test_split: str = DataModuleConfig.test_split,
    cache_dir: str = Config.cache_dir,
    seed: int = Config.seed,
) -> DatasetDict:
    """
    Load dataset from cache if available, otherwise download from Kaggle,
    process it, and save to cache.

    Returns:
        HuggingFace DatasetDict with train, valid, and test splits
    """
    cached_ds_dir = os.path.join(cache_dir, dataset_name)

    cached_ds_exists: bool = os.path.exists(cached_ds_dir)
    cached_ds_dir_is_empty: bool = len(os.listdir(cached_ds_dir)) == 0

    if cached_ds_exists and not cached_ds_dir_is_empty:
        print("Data cache exists. Loading from cache.")
        dataset = DatasetDict.load_from_disk(cached_ds_dir)
    else:
        print("Downloading dataset.")
        df_dict = _download_dataset_from_kaggle(
            dataset_name, dataset_paths, train_split, test_split
        )
        dataset = _clean_and_prepare_dataset(
            df_dict, label_list, train_split, test_split
        )
        print("Saving dataset to cache.")
        create_dirs(cached_ds_dir)
        dataset.save_to_disk(cached_ds_dir)

    return dataset


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


if __name__ == "__main__":
    datasets = load_dataset()
    for split_name, dataset in datasets.items():
        print(f"{split_name}: {len(dataset)} examples")
    print(f"Example features: {list(datasets['train'].features.keys())}")
