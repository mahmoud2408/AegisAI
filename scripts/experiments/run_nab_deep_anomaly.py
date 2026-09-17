"""CLI entrypoint for the Phase 4 NAB deep anomaly experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.ml.anomaly.nab_deep_experiment import (  # noqa: E402
    load_nab_deep_config,
    run_nab_deep_anomaly_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 NAB deep anomaly experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/nab_deep_anomaly.yaml"),
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
        "--max-series",
        type=int,
        default=None,
        help="Optional debug limit overriding config max_series.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_nab_deep_config(args.config)
    if args.max_series is not None:
        from dataclasses import replace

        config = replace(config, max_series=args.max_series)

    result = run_nab_deep_anomaly_experiment(
        root=args.root,
        output_root=args.output_root,
        run_id=args.run_id,
        config=config,
    )
    payload: dict[str, Any] = {
        "run_dir": str(result.run_dir),
        "dataset_summary": result.dataset_summary,
        "label_alignment": {
            key: value
            for key, value in result.label_alignment.items()
            if key.endswith("_count") or key in {"series_count", "observation_count"}
        },
        "global_metrics": result.metrics["global"],
        "computational_efficiency": result.metrics["computational_efficiency"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
