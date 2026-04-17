from __future__ import annotations

import inspect
from typing import Any

import torch.nn as nn
from transformers import Trainer, TrainingArguments

from cad_finetune.utils.logging import get_logger

_LOGGER = get_logger(__name__)


def _eval_strategy_training_args_kwarg(strategy: str) -> dict[str, str]:
    """Transformers 新 API 使用 eval_strategy；旧版仍用 evaluation_strategy。"""
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        return {"eval_strategy": strategy}
    return {"evaluation_strategy": strategy}


def _instantiate_training_arguments(**kwargs: Any) -> TrainingArguments:
    """只传入当前 transformers 版本里 ``TrainingArguments`` 实际支持的参数。

    v5 起会移除部分字段（如 ``save_safetensors``，因默认只保留 safetensors）；
    与旧版硬编码关键字相比，按 ``__init__`` 签名过滤可避免 Colab/本地版本不一致时反复报错。
    """
    params = inspect.signature(TrainingArguments.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        _LOGGER.info(
            "Skipping TrainingArguments keys not supported by this transformers build: %s",
            dropped,
        )
    return TrainingArguments(**filtered)


def trainer_tokenizer_kwarg(tokenizer: Any) -> dict[str, Any]:
    """Transformers v5+ 的 ``Trainer`` 使用 ``processing_class``；旧版使用 ``tokenizer``。"""
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if self.class_weights is None:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            w = self.class_weights.float().to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=w)
            num_classes = logits.shape[-1]
            loss = loss_fct(logits.float().view(-1, num_classes), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def _deepspeed_config_for_trainer(config: dict[str, Any]) -> str | dict[str, Any] | None:
    """让 DeepSpeed JSON 里的 fp16/bf16 与 TrainingArguments（如 shell 里 --fp16/--bf16）一致。

    配置加载时会把 deepspeed 文件读进 config['deepspeed']；这里就地同步后再把 dict 传给 Trainer，
    这样不必每次改磁盘上的 zero2.json。
    """
    training_cfg = config["training"]
    fp16 = bool(training_cfg.get("fp16", False))
    bf16 = bool(training_cfg.get("bf16", False))
    ds = config.get("deepspeed")
    if isinstance(ds, dict):
        if "fp16" in ds and isinstance(ds["fp16"], dict):
            ds["fp16"]["enabled"] = fp16
        if "bf16" in ds and isinstance(ds["bf16"], dict):
            ds["bf16"]["enabled"] = bf16
        return ds
    return config.get("deepspeed_config_path")


def build_training_arguments(
    config: dict[str, Any],
    has_eval_dataset: bool,
    *,
    attach_deepspeed: bool | None = None,
) -> TrainingArguments:
    training_cfg = config["training"]
    runtime_cfg = config.get("runtime", {})

    evaluation_strategy = training_cfg.get("evaluation_strategy", "no")
    if not has_eval_dataset:
        evaluation_strategy = "no"

    load_best_model_at_end = bool(training_cfg.get("load_best_model_at_end", False))
    if evaluation_strategy == "no":
        load_best_model_at_end = False

    report_to = runtime_cfg.get("report_to", [])
    deepspeed_arg: str | dict[str, Any] | None = None
    if attach_deepspeed is False:
        pass  # 独立 eval（python -m cli.eval）：勿传 deepspeed，否则 predict 会误走 ZeRO 推理并报错
    elif attach_deepspeed is True:
        deepspeed_arg = _deepspeed_config_for_trainer(config)
    elif runtime_cfg.get("launcher") == "deepspeed":
        deepspeed_arg = _deepspeed_config_for_trainer(config)

    ta: dict[str, Any] = {
        "output_dir": config.get("output_dir", "outputs/checkpoints/default"),
        "per_device_train_batch_size": training_cfg.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": training_cfg.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps", 1),
        "learning_rate": training_cfg.get("learning_rate", 5e-5),
        "weight_decay": training_cfg.get("weight_decay", 0.0),
        "optim": training_cfg.get("optim", "adamw_torch"),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "linear"),
        "num_train_epochs": training_cfg.get("num_train_epochs", 1),
        "logging_steps": training_cfg.get("logging_steps", 10),
        "logging_first_step": training_cfg.get("logging_first_step", False),
        "log_level": training_cfg.get("log_level", "passive"),
        "save_strategy": training_cfg.get("save_strategy", "steps"),
        "save_steps": training_cfg.get("save_steps", 100),
        "eval_steps": training_cfg.get("eval_steps", 100),
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "f1"),
        "greater_is_better": training_cfg.get("greater_is_better", True),
        "save_total_limit": training_cfg.get("save_total_limit", 2),
        "max_steps": training_cfg.get("max_steps", -1),
        "fp16": training_cfg.get("fp16", False),
        "bf16": training_cfg.get("bf16", False),
        "tf32": training_cfg.get("tf32", True),
        "remove_unused_columns": training_cfg.get("remove_unused_columns", False),
        "max_grad_norm": training_cfg.get("max_grad_norm", 1.0),
        "warmup_ratio": training_cfg.get("warmup_ratio", 0.0),
        "save_safetensors": training_cfg.get("save_safetensors", True),
        "gradient_checkpointing": runtime_cfg.get("gradient_checkpointing", False),
        "report_to": report_to,
        "seed": runtime_cfg.get("seed", 42),
        "dataloader_num_workers": runtime_cfg.get("dataloader_num_workers", 0),
        "deepspeed": deepspeed_arg,
        "label_names": ["labels"],
    }
    if config.get("logging_dir"):
        ta["logging_dir"] = config["logging_dir"]
    ta.update(_eval_strategy_training_args_kwarg(evaluation_strategy))
    return _instantiate_training_arguments(**ta)
