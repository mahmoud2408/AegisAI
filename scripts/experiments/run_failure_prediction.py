"""CLI entrypoint for Phase 7 failure prediction experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.ml.prediction.failure_prediction import (  # noqa: E402
    load_failure_prediction_config,
    run_failure_prediction_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 7 failure prediction experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/failure_prediction.yaml"),
        help="YAML experiment configuration.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override experiment output root.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional deterministic run id.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_failure_prediction_config(args.config)
    result = run_failure_prediction_experiment(
        root=args.root,
        output_root=args.output_root,
        run_id=args.run_id,
        config=config,
    )
    payload: dict[str, Any] = {
        "run_dir": str(result.run_dir),
        "best_models": result.metrics["best_models"],
        "elapsed_seconds": result.metrics["elapsed_seconds"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
