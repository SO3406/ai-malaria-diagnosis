#holds all the helper functions needed across the entire project
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1] # finds the project root in thie case ai-malaria-diagnosis


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]: # loads the configuration file
    path = project_path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    return config


def ensure_dir(path: str | Path) -> Path:
    directory = project_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
    return output_path
