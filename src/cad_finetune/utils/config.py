from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_reference(base_dir: Path, ref: str) -> Path:
    candidates = [
        (base_dir / ref).resolve(),
        (Path.cwd() / ref).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve config reference: {ref}")


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    experiment_cfg = _read_yaml(config_path)

    paths = experiment_cfg.pop("paths", {})
    merged_cfg = deepcopy(experiment_cfg)

    for section_name, ref in paths.items():
        resolved_path = _resolve_reference(config_path.parent, ref)
        if resolved_path.suffix in {".yaml", ".yml"}:
            section_cfg = _read_yaml(resolved_path)
        elif resolved_path.suffix == ".json":
            section_cfg = _read_json(resolved_path)
        else:
            raise ValueError(f"Unsupported config file type: {resolved_path}")

        if section_name == "deepspeed":
            merged_cfg["deepspeed"] = section_cfg
            merged_cfg["deepspeed_config_path"] = str(resolved_path)
        else:
            merged_cfg[section_name] = _deep_merge(section_cfg, merged_cfg.get(section_name, {}))
            merged_cfg[f"{section_name}_config_path"] = str(resolved_path)

    merged_cfg.setdefault("training", {})
    merged_cfg.setdefault("runtime", {})
    if merged_cfg["runtime"].get("launcher") is None:
        merged_cfg["runtime"]["launcher"] = (
            "deepspeed" if merged_cfg.get("deepspeed_config_path") else "python"
        )

    merged_cfg["config_path"] = str(config_path)
    return merged_cfg
