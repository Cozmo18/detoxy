import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class ToxyDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        text_col: str,
        label_cols: list[str],
        tokenizer: PreTrainedTokenizerFast,
        max_len: int,
    ) -> None:
        super().__init__()

        self.data = data
        self.text_col = text_col
        self.label_cols = label_cols
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = self.data.iloc[index][self.text_col]
        label = self.data.iloc[index][self.label_cols]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)  # type: ignore
        attention_mask = encoding["attention_mask"].squeeze(0)  # type: ignore
        token_type_ids = encoding["token_type_ids"].squeeze(0)  # type: ignore

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": torch.tensor(label, dtype=torch.float),
        }
