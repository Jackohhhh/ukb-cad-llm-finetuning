from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import Trainer, TrainerCallback

from cad_finetune.models import build_model_and_tokenizer, load_checkpoint_for_eval
from cad_finetune.tasks.classification.dataset import build_classification_datasets
from cad_finetune.tasks.classification.metrics import (
    build_compute_metrics,
    save_prediction_artifacts,
)
from cad_finetune.train.trainer import WeightedTrainer, build_training_arguments, trainer_tokenizer_kwarg
from cad_finetune.utils.logging import get_logger
from cad_finetune.utils.seed import set_global_seed


LOGGER = get_logger(__name__)


def _ensure_peft_adapters_record_base_model(output_dir: str | Path, base_model_name_or_path: str | None) -> None:
    """PEFT 保存时可能不写 base_model_name_or_path；补写后其它工具可直接从 adapter_config 读取基座 id。"""
    if not base_model_name_or_path:
        return
    root = Path(output_dir)
    if not root.is_dir():
        return
    for cfg_path in root.rglob("adapter_config.json"):
        try:
            with cfg_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("base_model_name_or_path"):
            continue
        data["base_model_name_or_path"] = base_model_name_or_path
        with cfg_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


class _AdapterBaseModelCallback(TrainerCallback):
    """每次保存 checkpoint 后，为当次目录下的 adapter_config.json 补全 base_model_name_or_path。"""

    def __init__(self, base_model_name_or_path: str) -> None:
        self._base = base_model_name_or_path

    def on_save(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        _ensure_peft_adapters_record_base_model(args.output_dir, self._base)
        return control


def _ensure_output_dirs(config: dict[str, Any]) -> None:
    Path(config.get("output_dir", "outputs/checkpoints/default")).mkdir(parents=True, exist_ok=True)
    Path(config.get("prediction_output_dir", "outputs/predictions/default")).mkdir(
        parents=True,
        exist_ok=True,
    )


def _build_trainer(config: dict[str, Any], model, tokenizer, data_module):
    training_args = build_training_arguments(config, has_eval_dataset=data_module.eval_dataset is not None)
    compute_metrics = build_compute_metrics(num_labels=config["task"].get("num_labels", 2))

    callbacks: list[TrainerCallback] = []
    base_id = config.get("model", {}).get("model_name_or_path")
    if base_id:
        callbacks.append(_AdapterBaseModelCallback(base_id))

    return WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module.train_dataset,
        eval_dataset=data_module.eval_dataset,
        data_collator=data_module.data_collator,
        compute_metrics=compute_metrics,
        class_weights=data_module.class_weights,
        callbacks=callbacks,
        **trainer_tokenizer_kwarg(tokenizer),
    )


def run_train(config: dict[str, Any]) -> None:
    _ensure_output_dirs(config)
    set_global_seed(config.get("runtime", {}).get("seed", 42))

    LOGGER.info("Loading model and tokenizer.")
    model, tokenizer = build_model_and_tokenizer(config)

    LOGGER.info("Preparing datasets.")
    data_module = build_classification_datasets(config, tokenizer)

    LOGGER.info("Building trainer.")
    trainer = _build_trainer(config, model, tokenizer, data_module)

    LOGGER.info("Starting training.")
    train_result = trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    trainer.save_state()
    _ensure_peft_adapters_record_base_model(
        config["output_dir"],
        config.get("model", {}).get("model_name_or_path"),
    )

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if data_module.eval_dataset is not None:
        LOGGER.info("Running validation.")
        eval_metrics = trainer.evaluate(eval_dataset=data_module.eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if data_module.test_dataset is not None and config.get("training", {}).get("run_test_after_train", True):
        LOGGER.info("Running test prediction.")
        prediction_output = trainer.predict(data_module.test_dataset)
        save_prediction_artifacts(
            output_dir=config.get("prediction_output_dir", "outputs/predictions/default"),
            prediction_output=prediction_output,
            raw_dataset=data_module.raw_test_dataset,
            label_column=config["task"].get("label_column", "output"),
        )


def run_eval(config: dict[str, Any], checkpoint_path: str) -> None:
    _ensure_output_dirs(config)
    set_global_seed(config.get("runtime", {}).get("seed", 42))

    LOGGER.info("Loading checkpoint for evaluation.")
    model, tokenizer = load_checkpoint_for_eval(config, checkpoint_path)
    data_module = build_classification_datasets(config, tokenizer)

    training_args = build_training_arguments(
        config,
        has_eval_dataset=data_module.test_dataset is not None,
        attach_deepspeed=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_module.data_collator,
        compute_metrics=build_compute_metrics(num_labels=config["task"].get("num_labels", 2)),
        **trainer_tokenizer_kwarg(tokenizer),
    )

    if data_module.test_dataset is None:
        raise ValueError("The dataset config does not define test_file, so evaluation cannot run.")

    prediction_output = trainer.predict(data_module.test_dataset)
    save_prediction_artifacts(
        output_dir=config.get("prediction_output_dir", "outputs/predictions/default"),
        prediction_output=prediction_output,
        raw_dataset=data_module.raw_test_dataset,
        label_column=config["task"].get("label_column", "output"),
    )
