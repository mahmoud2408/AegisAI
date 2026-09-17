"""CLI entrypoint for the Phase 6 SMD multivariate anomaly experiment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.ml.anomaly.smd_multivariate import (  # noqa: E402
    load_smd_multivariate_config,
    run_smd_multivariate_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 6 SMD multivariate anomaly experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/smd_multivariate.yaml"),
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
    parser.add_argument(
        "--max-machines",
        type=int,
        default=None,
        help="Optional debug limit overriding config max_machines.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_smd_multivariate_config(args.config)
    if args.max_machines is not None:
        config = replace(config, max_machines=args.max_machines)

    result = run_smd_multivariate_experiment(
        root=args.root,
        output_root=args.output_root,
        run_id=args.run_id,
        config=config,
    )
    payload: dict[str, Any] = {
        "run_dir": str(result.run_dir),
        "dataset_audit": {
            key: result.dataset_audit[key]
            for key in (
                "machine_count",
                "metric_count",
                "train_rows",
                "test_rows",
                "positive_test_labels",
                "positive_label_rate",
                "anomaly_segments",
            )
        },
        "best_model": result.metrics["best_model"],
        "multivariate_benefit": result.metrics["multivariate_benefit"],
        "elapsed_seconds": result.metrics["elapsed_seconds"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
