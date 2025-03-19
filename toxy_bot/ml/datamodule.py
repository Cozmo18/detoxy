import pytorch_lightning as pl


class ToxyDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
