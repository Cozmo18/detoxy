from pathlib import Path

import datasets
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

from model.config import Config, DataConfig

DATA_DIR: Path = Config().data_dir
CLASSES: list[str] = DataConfig().classes


class JigsawData(Dataset):
    """Dataloader for the original Jigsaw Toxic Comment Classification Challenge.

    Source:
        https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    """

    def __init__(
        self,
        train_csv_file: str | Path = DATA_DIR / "train.csv",
        test_csv_file: str | Path = DATA_DIR / "test.csv",
        classes: list[str] = CLASSES,
        *,
        train: bool = True,
        add_test_labels: bool = True,
    ) -> None:
        self.train = train
        self.add_test_labels = add_test_labels
        self.classes = classes

        if train:
            self.dataframe = pd.read_csv(train_csv_file)
        else:
            self.dataframe = self.load_test_df(
                test_csv_file, add_labels=add_test_labels
            )

        self.dataset = datasets.Dataset.from_pandas(self.dataframe)

    def load_test_df(
        self,
        test_csv_file: str | Path,
        *,
        add_labels: bool = False
    ) -> pd.DataFrame:
        test_df = pd.read_csv(test_csv_file)
        if add_labels:
            test_labels_df = pd.read_csv(str(test_csv_file)[:-4] + "_labels.csv")
            test_df = test_df.merge(test_labels_df, on="id", validate="1:1")
        return test_df

    def __getitem__(self, index: int) -> dict:
        row = self.dataset[index]
        text = row["comment_text"]
        labels = {label: value for label, value in row.items() if label in self.classes}

        return {"text": text, "labels": torch.FloatTensor(list(labels.values()))}


if __name__ == "__main__":
    dataset = JigsawData(train=True)

    sample = dataset[0]
    print(sample)
