from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def tokenize_text(
    batch: str | list[str] | dict,
    *,
    model_name: str,
    max_seq_length: int,
    cache_dir: str | Path,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, cache_dir=cache_dir
    )
    # assume text has a "text" key in dict
    text = (
        batch["text"] if isinstance(batch, dict) else batch
    )  # Allow for inference input as raw text
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )


def combine_labels(batch: dict, labels: list[str]) -> list:
    batch_size = len(batch[labels[0]])
    num_labels = len(labels)

    labels_batch = {k: batch[k] for k in batch if k in labels}
    labels_matrix = np.zeros((batch_size, num_labels))

    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    return labels_matrix.tolist()
