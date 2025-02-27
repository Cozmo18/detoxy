from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]


def load_config():
    config_path = ROOT_DIR / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
