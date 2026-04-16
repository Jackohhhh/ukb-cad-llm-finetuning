from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.utils.class_weight import compute_class_weight

from cad_finetune.data.collators import build_data_collator


@dataclass
class ClassificationDataModule:
    train_dataset: Dataset
    eval_dataset: Dataset | None
    test_dataset: Dataset | None
    data_collator: Any
    class_weights: torch.Tensor | None
    raw_test_dataset: Dataset | None


def _resolve_data_file(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    candidate = Path.cwd() / path
    return str(candidate.resolve())


def _load_json_dataset(path_str: str) -> Dataset:
    return load_dataset("json", data_files=_resolve_data_file(path_str), split="train")


def _oversample_label(
    dataset: Dataset,
    label_column: str,
    target_label: int,
    repeat_times: int,
    shuffle_seed: int,
) -> Dataset:
    if repeat_times <= 1:
        return dataset

    positive_dataset = dataset.filter(lambda row: int(row[label_column]) == int(target_label))
    negative_dataset = dataset.filter(lambda row: int(row[label_column]) != int(target_label))

    if len(positive_dataset) == 0:
        raise ValueError(
            f"Oversampling was enabled, but no samples were found for label={target_label}."
        )

    upsampled_positive = concatenate_datasets([positive_dataset] * repeat_times)
    return concatenate_datasets([negative_dataset, upsampled_positive]).shuffle(seed=shuffle_seed)


def _compute_class_weights(
    train_dataset: Dataset,
    label_column: str,
    num_labels: int,
    class_weight_cfg: dict[str, Any] | None,
) -> torch.Tensor | None:
    if not class_weight_cfg:
        return None

    mode = class_weight_cfg.get("mode", "none")
    if mode == "none":
        return None
    if mode == "manual":
        return torch.tensor(class_weight_cfg["values"], dtype=torch.float32)
    if mode == "balanced":
        labels = np.array([int(row[label_column]) for row in train_dataset], dtype=np.int64)
        classes = np.arange(num_labels)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
        return torch.tensor(class_weights, dtype=torch.float32)
    raise ValueError(f"Unsupported class_weights mode: {mode}")


def build_classification_datasets(config: dict[str, Any], tokenizer) -> ClassificationDataModule:
    dataset_cfg = config["dataset"]
    task_cfg = config["task"]

    label_column = task_cfg.get("label_column", "output")
    input_column = task_cfg.get("input_column", "input")

    train_dataset = _load_json_dataset(dataset_cfg["train_file"])

    validation_file = dataset_cfg.get("validation_file")
    if validation_file:
        eval_dataset = _load_json_dataset(validation_file)
    else:
        split_dataset = train_dataset.train_test_split(
            test_size=dataset_cfg.get("validation_split", 0.2),
            seed=dataset_cfg.get("split_seed", 42),
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    test_dataset = None
    raw_test_dataset = None
    if dataset_cfg.get("test_file"):
        raw_test_dataset = _load_json_dataset(dataset_cfg["test_file"])
        test_dataset = raw_test_dataset

    oversampling_cfg = dataset_cfg.get("oversampling", {})
    if oversampling_cfg.get("enabled", False):
        train_dataset = _oversample_label(
            dataset=train_dataset,
            label_column=label_column,
            target_label=oversampling_cfg.get("target_label", 1),
            repeat_times=oversampling_cfg.get("repeat_times", 1),
            shuffle_seed=dataset_cfg.get("shuffle_seed", 42),
        )

    class_weights = _compute_class_weights(
        train_dataset=train_dataset,
        label_column=label_column,
        num_labels=task_cfg.get("num_labels", 2),
        class_weight_cfg=dataset_cfg.get("class_weights"),
    )

    def preprocess_function(examples: dict[str, list[Any]]) -> dict[str, Any]:
        inputs = [str(doc) for doc in examples[input_column]]
        tokenized = tokenizer(
            inputs,
            max_length=task_cfg.get("max_length", 1024),
            truncation=task_cfg.get("truncation", True),
        )
        tokenized["labels"] = [int(value) for value in examples[label_column]]
        return tokenized

    def tokenize_dataset(dataset: Dataset, shuffle_seed: int | None) -> Dataset:
        tokenized = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        if shuffle_seed is not None:
            tokenized = tokenized.shuffle(seed=shuffle_seed)
        return tokenized

    tokenized_train = tokenize_dataset(train_dataset, dataset_cfg.get("train_shuffle_seed"))
    tokenized_eval = tokenize_dataset(eval_dataset, dataset_cfg.get("eval_shuffle_seed"))
    tokenized_test = (
        tokenize_dataset(test_dataset, None)
        if test_dataset is not None
        else None
    )

    return ClassificationDataModule(
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        test_dataset=tokenized_test,
        data_collator=build_data_collator(tokenizer),
        class_weights=class_weights,
        raw_test_dataset=raw_test_dataset,
    )
