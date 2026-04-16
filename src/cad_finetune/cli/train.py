from __future__ import annotations

import argparse
from pathlib import Path

from cad_finetune.cli.overrides import apply_experiment_cli_overrides, register_experiment_override_args
from cad_finetune.train.runner import run_train
from cad_finetune.utils.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a finetuning experiment. Base settings come from --config YAML; "
        "optional flags override (typically from shell scripts).",
    )
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file.")
    register_experiment_override_args(parser)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the test-set prediction stage after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(Path(args.config))
    apply_experiment_cli_overrides(config, args)
    if args.skip_test:
        config.setdefault("training", {})["run_test_after_train"] = False
    run_train(config)


if __name__ == "__main__":
    main()
