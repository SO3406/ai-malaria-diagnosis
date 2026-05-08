from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .data_loader import build_datasets
from .utils import ensure_dir, load_config, project_path, write_json


def evaluate(
    config_path: str | Path = "config/config.yaml",
    model_path: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    _, val_ds, class_names = build_datasets(config)

    if model_path is None:
        model_path = Path(config["model"]["output_dir"]) / config["model"]["checkpoint_name"]
    model = tf.keras.models.load_model(project_path(model_path))

    probabilities = model.predict(val_ds).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    labels = np.concatenate([y.numpy().ravel().astype(int) for _, y in val_ds])

    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions).tolist()

    reports_dir = ensure_dir(config["reports"]["output_dir"])
    metrics_path = reports_dir / config["reports"]["metrics_name"]
    write_json(
        {
            "class_names": class_names,
            "classification_report": report,
            "confusion_matrix": matrix,
        },
        metrics_path,
    )
    return metrics_path


if __name__ == "__main__":
    evaluate()
