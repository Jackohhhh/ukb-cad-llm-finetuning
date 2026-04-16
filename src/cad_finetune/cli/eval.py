from __future__ import annotations

import argparse
from pathlib import Path

from cad_finetune.cli.overrides import apply_experiment_cli_overrides, register_experiment_override_args
from cad_finetune.train.runner import run_eval
from cad_finetune.utils.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a finetuned checkpoint. Uses the same optional overrides as train "
        "(model, data, runtime, training args used for Trainer/predict batch sizes).",
    )
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint or adapter path.")
    register_experiment_override_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(Path(args.config))
    apply_experiment_cli_overrides(config, args)
    run_eval(config, args.checkpoint)


if __name__ == "__main__":
    main()
