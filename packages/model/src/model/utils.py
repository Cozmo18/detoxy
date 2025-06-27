from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def move_to(obj: torch.Tensor | dict | list, device: str) -> torch.Tensor | dict | list:
    """Utility function to move objects of different types containing tensors
    to a specified device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)

    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    if isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res

    msg = "Invalid type"
    raise TypeError(msg)


def create_dirs(dirs: list[str | Path]) -> None:
    for d in dirs:
        path = Path(d) if isinstance(d, str) else d
        if not path.is_dir():
            path.mkdir()


def get_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    cache_dir: str | Path,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
        problem_type="multi_label_classification",
    )
    return tokenizer, model
