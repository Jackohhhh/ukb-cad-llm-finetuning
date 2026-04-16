from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_compute_metrics(num_labels: int = 2):
    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        predictions = np.argmax(logits, axis=-1)

        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0, average="binary"),
            "recall": recall_score(labels, predictions, zero_division=0, average="binary"),
            "f1": f1_score(labels, predictions, zero_division=0, average="binary"),
        }

        if num_labels == 2:
            probabilities = _softmax(logits)[:, 1]
            try:
                metrics["auc"] = roc_auc_score(labels, probabilities)
            except ValueError:
                metrics["auc"] = float("nan")

        return metrics

    return compute_metrics


def save_prediction_artifacts(
    output_dir: str,
    prediction_output,
    raw_dataset=None,
    label_column: str = "output",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logits = np.asarray(prediction_output.predictions)
    labels = np.asarray(prediction_output.label_ids)
    probabilities = _softmax(logits)
    predictions = np.argmax(logits, axis=-1)
    positive_probabilities = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

    summary = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0, average="binary"),
        "recall": recall_score(labels, predictions, zero_division=0, average="binary"),
        "f1": f1_score(labels, predictions, zero_division=0, average="binary"),
        "classification_report": classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }
    try:
        summary["auc"] = roc_auc_score(labels, positive_probabilities)
    except ValueError:
        summary["auc"] = float("nan")

    metrics_file = output_path / "metrics.json"
    metrics_file.write_text(
        json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        row = {
            "index": index,
            "label": int(labels[index]),
            "prediction": int(prediction),
            "positive_probability": float(positive_probabilities[index]),
        }
        if raw_dataset is not None:
            original_row = raw_dataset[index]
            for key, value in original_row.items():
                if key == label_column:
                    continue
                row[key] = value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
        rows.append(row)

    prediction_file = output_path / "predictions.jsonl"
    with prediction_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")
