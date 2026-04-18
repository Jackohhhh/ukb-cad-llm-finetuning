"""Shared CLI overrides for train/eval (hyperparameters live in shell scripts, not configs/training)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def apply_experiment_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.model_name_or_path:
        config.setdefault("model", {})["model_name_or_path"] = args.model_name_or_path
    if args.tokenizer_name_or_path:
        config.setdefault("model", {})["tokenizer_name_or_path"] = args.tokenizer_name_or_path
    elif args.model_name_or_path:
        config.setdefault("model", {})["tokenizer_name_or_path"] = args.model_name_or_path
    if args.num_labels is not None:
        config.setdefault("task", {})["num_labels"] = args.num_labels
        config.setdefault("model", {})["num_labels"] = args.num_labels
    if args.max_length is not None:
        config.setdefault("task", {})["max_length"] = args.max_length
    if args.input_column:
        config.setdefault("task", {})["input_column"] = args.input_column
    if args.label_column:
        config.setdefault("task", {})["label_column"] = args.label_column

    if args.train_file:
        config.setdefault("dataset", {})["train_file"] = args.train_file
    if args.validation_file is not None:
        config.setdefault("dataset", {})["validation_file"] = (
            None if args.validation_file.strip() == "" else args.validation_file
        )
    if args.test_file:
        config.setdefault("dataset", {})["test_file"] = args.test_file

    if getattr(args, "experiment_name", None):
        config["experiment_name"] = args.experiment_name
    if getattr(args, "output_dir", None):
        config["output_dir"] = args.output_dir
    if getattr(args, "prediction_output_dir", None):
        config["prediction_output_dir"] = args.prediction_output_dir
    if args.logging_dir:
        config["logging_dir"] = args.logging_dir

    if args.launcher:
        config.setdefault("runtime", {})["launcher"] = args.launcher
    if args.seed is not None:
        config.setdefault("runtime", {})["seed"] = args.seed
    if args.gradient_checkpointing is not None:
        config.setdefault("runtime", {})["gradient_checkpointing"] = args.gradient_checkpointing == "true"
    if args.dataloader_num_workers is not None:
        config.setdefault("runtime", {})["dataloader_num_workers"] = args.dataloader_num_workers
    if args.report_to is not None:
        if args.report_to.lower() in {"", "none", "[]"}:
            config.setdefault("runtime", {})["report_to"] = []
        else:
            config.setdefault("runtime", {})["report_to"] = [args.report_to]

    if args.deepspeed:
        ds_path = Path(args.deepspeed).expanduser().resolve()
        with ds_path.open("r", encoding="utf-8") as handle:
            config["deepspeed"] = json.load(handle)
        config["deepspeed_config_path"] = str(ds_path)

    t = config.setdefault("training", {})
    if args.num_train_epochs is not None:
        t["num_train_epochs"] = args.num_train_epochs
    if args.per_device_train_batch_size is not None:
        t["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.per_device_eval_batch_size is not None:
        t["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    if args.gradient_accumulation_steps is not None:
        t["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        t["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        t["weight_decay"] = args.weight_decay
    if args.warmup_ratio is not None:
        t["warmup_ratio"] = args.warmup_ratio
    if args.lr_scheduler_type:
        t["lr_scheduler_type"] = args.lr_scheduler_type
    if args.logging_steps is not None:
        t["logging_steps"] = args.logging_steps
    if args.save_strategy:
        t["save_strategy"] = args.save_strategy
    if args.save_steps is not None:
        t["save_steps"] = args.save_steps
    if args.evaluation_strategy:
        t["evaluation_strategy"] = args.evaluation_strategy
    if args.eval_steps is not None:
        t["eval_steps"] = args.eval_steps
    if args.save_total_limit is not None:
        t["save_total_limit"] = args.save_total_limit
    if args.max_steps is not None:
        t["max_steps"] = args.max_steps
    if args.load_best_model_at_end is not None:
        t["load_best_model_at_end"] = args.load_best_model_at_end == "true"
    if args.metric_for_best_model:
        t["metric_for_best_model"] = args.metric_for_best_model
    if args.greater_is_better is not None:
        t["greater_is_better"] = args.greater_is_better == "true"
    if args.max_grad_norm is not None:
        t["max_grad_norm"] = args.max_grad_norm
    if args.optim:
        t["optim"] = args.optim
    if args.logging_first_step is not None:
        t["logging_first_step"] = args.logging_first_step == "true"
    if args.log_level:
        t["log_level"] = args.log_level

    if args.fp16:
        t["fp16"] = True
        t["bf16"] = False
    if args.bf16:
        t["bf16"] = True
        t["fp16"] = False
    if args.tf32 is not None:
        t["tf32"] = args.tf32 == "true"

    m = config.setdefault("model", {})
    lora = m.setdefault("lora", {})
    if args.lora_enable is not None:
        lora["enabled"] = args.lora_enable == "true"
    if args.lora_r is not None:
        lora["r"] = args.lora_r
    if args.lora_alpha is not None:
        lora["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        lora["lora_dropout"] = args.lora_dropout

    if args.load_in_4bit is not None:
        m["load_in_4bit"] = args.load_in_4bit == "true"
        if args.load_in_4bit == "true":
            m["load_in_8bit"] = False
    if args.load_in_8bit is not None:
        m["load_in_8bit"] = args.load_in_8bit == "true"
        if args.load_in_8bit == "true":
            m["load_in_4bit"] = False

    if args.trust_remote_code is not None:
        m["trust_remote_code"] = args.trust_remote_code == "true"
    if args.attn_implementation is not None:
        if args.attn_implementation.strip() == "":
            m.pop("attn_implementation", None)
        else:
            m["attn_implementation"] = args.attn_implementation.strip()


def register_experiment_override_args(
    parser: argparse.ArgumentParser,
    *,
    include_run_paths: bool = True,
) -> None:
    m = parser.add_argument_group("model / data overrides")
    m.add_argument("--model-name-or-path", default=None, help="HF id or local path for weights.")
    m.add_argument("--tokenizer-name-or-path", default=None, help="Tokenizer id or path (defaults follow model YAML).")
    m.add_argument("--num-labels", type=int, default=None)
    m.add_argument("--max-length", type=int, default=None)
    m.add_argument("--input-column", default=None)
    m.add_argument("--label-column", default=None)
    m.add_argument("--train-file", default=None, help="Training JSON path (relative paths resolve from cwd).")
    m.add_argument("--validation-file", default=None, help="Optional fixed validation JSON; use empty string to clear.")
    m.add_argument("--test-file", default=None)

    o = parser.add_argument_group("run paths")
    if include_run_paths:
        o.add_argument("--experiment-name", default=None)
        o.add_argument("--output-dir", default=None)
        o.add_argument("--prediction-output-dir", default=None)
    o.add_argument("--logging-dir", default=None)

    r = parser.add_argument_group("runtime / launcher")
    r.add_argument("--launcher", default=None, help="e.g. deepspeed or python (should match how you invoke training).")
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--gradient-checkpointing", choices=["true", "false"], default=None)
    r.add_argument("--dataloader-num-workers", type=int, default=None)
    r.add_argument(
        "--report-to",
        default=None,
        help='Integration name, e.g. "wandb", or "none" to disable.',
    )
    r.add_argument("--deepspeed", default=None, help="Path to DeepSpeed JSON (overrides experiment paths.deepspeed).")

    t = parser.add_argument_group("training (Transformers TrainingArguments)")
    t.add_argument("--num-train-epochs", type=float, default=None)
    t.add_argument("--per-device-train-batch-size", type=int, default=None)
    t.add_argument("--per-device-eval-batch-size", type=int, default=None)
    t.add_argument("--gradient-accumulation-steps", type=int, default=None)
    t.add_argument("--learning-rate", type=float, default=None)
    t.add_argument("--weight-decay", type=float, default=None)
    t.add_argument("--warmup-ratio", type=float, default=None)
    t.add_argument("--lr-scheduler-type", default=None)
    t.add_argument("--logging-steps", type=int, default=None)
    t.add_argument("--logging-first-step", choices=["true", "false"], default=None)
    t.add_argument("--log-level", default=None, help="e.g. info, passive (Transformers log level).")
    t.add_argument("--save-strategy", default=None)
    t.add_argument("--save-steps", type=int, default=None)
    t.add_argument("--evaluation-strategy", default=None)
    t.add_argument("--eval-steps", type=int, default=None)
    t.add_argument("--save-total-limit", type=int, default=None)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--load-best-model-at-end", choices=["true", "false"], default=None)
    t.add_argument("--metric-for-best-model", default=None)
    t.add_argument("--greater-is-better", choices=["true", "false"], default=None)
    t.add_argument("--max-grad-norm", type=float, default=None)
    t.add_argument("--optim", default=None)
    t.add_argument("--fp16", action="store_true", help="Force fp16 on (and bf16 off).")
    t.add_argument("--bf16", action="store_true", help="Force bf16 on (and fp16 off).")
    t.add_argument(
        "--tf32",
        choices=["true", "false"],
        default=None,
        help="TF32 for matmul on Ampere+ (default in TrainingArguments: on unless overridden).",
    )

    l = parser.add_argument_group("LoRA / quantization")
    l.add_argument("--lora-enable", choices=["true", "false"], default=None)
    l.add_argument("--lora-r", type=int, default=None)
    l.add_argument("--lora-alpha", type=int, default=None)
    l.add_argument("--lora-dropout", type=float, default=None)
    l.add_argument("--load-in-4bit", choices=["true", "false"], default=None)
    l.add_argument("--load-in-8bit", choices=["true", "false"], default=None)
    l.add_argument("--trust-remote-code", choices=["true", "false"], default=None)
    l.add_argument(
        "--attn-implementation",
        default=None,
        help='Attention 后端：flash_attention_2、sdpa、eager；传空字符串 "" 则清除 YAML 中的设置。',
    )
