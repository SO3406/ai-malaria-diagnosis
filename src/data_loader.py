from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf

from .utils import project_path, set_seed


AUTOTUNE = tf.data.AUTOTUNE # automatically optimize the data loading process


def validate_data_dir(data_dir: str | Path) -> Path:
    path = project_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {path}. Expected class folders such as "
            "'Parasitized' and 'Uninfected' inside it."
        )
    return path


def build_datasets(config: dict[str, Any]) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    data_config = config["data"]
    training_config = config["training"]

    data_dir = validate_data_dir(data_config["raw_dir"])
    image_size = tuple(training_config["image_size"])
    batch_size = int(training_config["batch_size"])
    seed = int(data_config.get("seed", 42))
    validation_split = 1.0 - float(data_config.get("train_split", 0.8))

    set_seed(seed)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
    )

    class_names = train_ds.class_names
    return optimize_dataset(train_ds, shuffle=True), optimize_dataset(val_ds), class_names


def optimize_dataset(dataset: tf.data.Dataset, shuffle: bool = False) -> tf.data.Dataset:
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset.prefetch(AUTOTUNE)
