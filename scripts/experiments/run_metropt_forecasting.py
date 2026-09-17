"""Run the Phase 8 MetroPT-3 forecasting experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.forecasting.metropt import (  # noqa: E402
    load_metropt_forecasting_config,
    run_metropt_forecasting_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 8 MetroPT forecasting experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/metropt_forecasting.yaml"),
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional deterministic run id for artifact paths.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root. Defaults to experiments/forecasting/metropt.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_metropt_forecasting_config(config_path)
    result = run_metropt_forecasting_experiment(
        root=root,
        output_root=args.output_root,
        run_id=args.run_id,
        config=config,
    )
    payload: dict[str, Any] = {
        "run_dir": str(result.run_dir),
        "best_validation_global": result.metrics["best_validation_global"],
        "validation_selected_test_global": result.metrics["validation_selected_test_global"],
        "best_test_global": result.metrics["best_test_global"],
        "elapsed_seconds": result.metrics["elapsed_seconds"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
