import pytorch_lightning as pl

from toxy_bot.ml.config import Config, DataModuleConfig, ModuleConfig
from toxy_bot.ml.dataset import load_dataset


class ToxyTokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str = ModuleConfig.model_name,
        num_labels: int = DataModuleConfig.num_labels,
        columns: list = ["input_ids", "attention_mask", "label"],
        train_size: float = DataModuleConfig.train_size,
        batch_size: int = DataModuleConfig.batch_size,
        train_split: str = DataModuleConfig.train_split,
        valid_split: str = DataModuleConfig.valid_split,
        test_split: str = DataModuleConfig.test_split,
        seed: int = Config.seed,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.columns = columns
        self.train_size = train_size
        self.batch_size = batch_size
        self.train_size = train_size
        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split
        self.seed = seed

        self.dataset = None

    def prepare_data(self) -> None:
        if self.dataset is None:
            ds = load_dataset()
            train_ds, valid_ds = (
                ds[self.train_split]
                .train_test_split(
                    train_size=self.train_size,
                    seed=self.seed,
                )
                .values()
            )
            ds[self.train_split] = train_ds
            ds[self.valid_split] = valid_ds

            self.dataset = ds
            print("Created validation split.")


if __name__ == "__main__":
    dm = ToxyTokenizerDataModule()
    dm.prepare_data()
    ds = dm.dataset
    print(ds)
