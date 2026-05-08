from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf

from .data_loader import build_datasets
from .model import build_cnn
from .utils import ensure_dir, load_config, project_path, set_seed, write_json


def build_callbacks(config: dict[str, Any]) -> list[tf.keras.callbacks.Callback]:
    model_dir = ensure_dir(config["model"]["output_dir"])
    checkpoint_path = model_dir / config["model"]["checkpoint_name"]

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
        ),
    ]


def train(config_path: str | Path = "config/config.yaml") -> Path:
    config = load_config(config_path)
    set_seed(int(config["data"].get("seed", 42)))

    train_ds, val_ds, class_names = build_datasets(config)
    training_config = config["training"]
    model = build_cnn(
        image_size=training_config["image_size"],
        learning_rate=float(training_config["learning_rate"]),
        dropout_rate=float(training_config["dropout_rate"]),
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(training_config["epochs"]),
        callbacks=build_callbacks(config),
    )

    model_dir = ensure_dir(config["model"]["output_dir"]) # make the function to append new traiing mtarix instead of overwritting or add a timestamp
    final_model_path = model_dir / config["model"]["final_name"]
    model.save(final_model_path)

    reports_dir = ensure_dir(config["reports"]["output_dir"])
    history_path = reports_dir / config["reports"]["history_name"]
    write_json(
        {
            "class_names": class_names,
            "history": history.history,
            "final_model": str(final_model_path.relative_to(project_path("."))),
        },
        history_path,
    )
    return final_model_path


if __name__ == "__main__":
    train()
