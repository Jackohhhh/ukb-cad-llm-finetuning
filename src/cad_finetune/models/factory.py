from __future__ import annotations

import os
from typing import Any

import torch
from peft import (
    AutoPeftModelForSequenceClassification,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from cad_finetune.models.sequence_classifier import BackboneForSequenceClassification


def _resolve_dtype(dtype_name: str | None) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.float16


def _build_quantization_config(model_cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    if model_cfg.get("load_in_4bit"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_resolve_dtype(model_cfg.get("bnb_4bit_compute_dtype")),
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        )
    if model_cfg.get("load_in_8bit"):
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _resolve_target_modules(model_cfg: dict[str, Any]) -> list[str]:
    configured = model_cfg.get("lora", {}).get("target_modules")
    if configured:
        return configured
    raise ValueError(
        "LoRA is enabled but `lora.target_modules` is missing in the model YAML "
        "(configs/models/*.yaml). Add the target module names there.",
    )


def _resolve_modules_to_save(model_cfg: dict[str, Any]) -> list[str] | None:
    configured = model_cfg.get("lora", {}).get("modules_to_save")
    if configured:
        return configured
    return None


def _resolve_device_map(model_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> Any:
    if model_cfg.get("device_map") is not None:
        return model_cfg.get("device_map")
    if model_cfg.get("load_in_4bit") or model_cfg.get("load_in_8bit"):
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return {"": int(local_rank)}
        if runtime_cfg.get("launcher") == "python":
            return "auto"
    return None


def _make_model_kwargs(model_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
        "cache_dir": model_cfg.get("cache_dir"),
    }
    attn = model_cfg.get("attn_implementation")
    if attn:
        kwargs["attn_implementation"] = attn
    quantization_config = _build_quantization_config(model_cfg)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        device_map = _resolve_device_map(model_cfg, runtime_cfg)
        if device_map is not None:
            kwargs["device_map"] = device_map
    return kwargs


def _make_custom_sequence_classifier(model_cfg: dict[str, Any], runtime_cfg: dict[str, Any]):
    backbone = AutoModel.from_pretrained(
        model_cfg["model_name_or_path"],
        **_make_model_kwargs(model_cfg, runtime_cfg),
    )
    return BackboneForSequenceClassification(
        backbone=backbone,
        num_labels=model_cfg.get("num_labels", 2),
        dropout=model_cfg.get("custom_head_dropout", 0.0),
    )


def _enable_trainable_head_parameters(model, model_cfg: dict[str, Any]) -> None:
    patterns = model_cfg.get("head_trainable_patterns", ["classifier", "score", "head"])
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = True


def build_model_and_tokenizer(config: dict[str, Any]):
    model_cfg = config["model"]
    runtime_cfg = config.get("runtime", {})

    tokenizer_name = model_cfg.get("tokenizer_name_or_path") or model_cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=model_cfg.get("cache_dir"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    tokenizer.padding_side = model_cfg.get("padding_side", "right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_cfg.get("use_custom_binary_head"):
        model = _make_custom_sequence_classifier(model_cfg, runtime_cfg)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["model_name_or_path"],
            num_labels=model_cfg.get("num_labels", 2),
            **_make_model_kwargs(model_cfg, runtime_cfg),
        )

    if hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if model_cfg.get("load_in_4bit") or model_cfg.get("load_in_8bit"):
        model = prepare_model_for_kbit_training(model)

    if config.get("runtime", {}).get("gradient_checkpointing") and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()

    lora_cfg = model_cfg.get("lora", {})
    if lora_cfg.get("enabled", False):
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=_resolve_target_modules(model_cfg),
            lora_dropout=lora_cfg.get("lora_dropout", 0.0),
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.SEQ_CLS,
            modules_to_save=_resolve_modules_to_save(model_cfg),
        )
        model = get_peft_model(model, peft_config)
        _enable_trainable_head_parameters(model, model_cfg)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    return model, tokenizer


def load_checkpoint_for_eval(config: dict[str, Any], checkpoint_path: str):
    model_cfg = config["model"]
    fallback_tokenizer_source = model_cfg.get("tokenizer_name_or_path") or model_cfg["model_name_or_path"]
    tok_kwargs = {
        "cache_dir": model_cfg.get("cache_dir"),
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, **tok_kwargs)
    except OSError:
        # 部分 checkpoint 目录未保存 tokenizer 文件时，回退到模型配置里的 Hub id / 本地路径
        tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer_source, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_cfg.get("padding_side", "right")

    try:
        model = AutoPeftModelForSequenceClassification.from_pretrained(checkpoint_path)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=model_cfg.get("num_labels", 2),
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )

    if hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
