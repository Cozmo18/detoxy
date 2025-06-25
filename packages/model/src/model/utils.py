import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel


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

    raise TypeError("Invalid type")


def get_model_and_tokenizer(
    model_name: str, num_labels: int
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type="multi_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return model, tokenizer
