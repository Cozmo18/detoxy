from pathlib import Path

DATASET_HANDLE = "julian3833/jigsaw-toxic-comment-classification-challenge"
DATASET_FEATURES = ["comment_text"]
DATASET_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


ROOT_DIR: Path = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TENSORFLOW_DATA_DIR = DATA_DIR / "tensorflow"
