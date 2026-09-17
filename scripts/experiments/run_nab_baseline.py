from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.features.timeseries import CausalRollingFeatureConfig  # noqa: E402
from aegis_ai.ml.anomaly.baselines import RollingZScoreBaselineConfig  # noqa: E402
from aegis_ai.ml.anomaly.isolation_forest import IsolationForestConfig  # noqa: E402
from aegis_ai.ml.anomaly.nab_experiment import (  # noqa: E402
    NabExperimentConfig,
    run_nab_baseline_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 3 NAB anomaly baseline.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "anomaly" / "nab" / "isolation_forest",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--short-window", type=int, default=5)
    parser.add_argument("--long-window", type=int, default=20)
    parser.add_argument("--trend-window", type=int, default=5)
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-samples", default="auto")
    parser.add_argument("--contamination", default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = NabExperimentConfig(
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        threshold_quantile=args.threshold_quantile,
        feature_config=CausalRollingFeatureConfig(
            short_window=args.short_window,
            long_window=args.long_window,
            trend_window=args.trend_window,
            min_periods=args.min_periods,
        ),
        zscore_config=RollingZScoreBaselineConfig(
            threshold_quantile=args.threshold_quantile,
        ),
        isolation_forest_config=IsolationForestConfig(
            n_estimators=args.n_estimators,
            max_samples=_parse_max_samples(args.max_samples),
            contamination=_parse_contamination(args.contamination),
            random_state=args.random_state,
        ),
    )
    result = run_nab_baseline_experiment(
        root=args.root,
        output_root=args.output_root,
        run_id=args.run_id,
        config=config,
    )
    payload = {
        "run_dir": str(result.run_dir),
        "config": {
            "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction,
            "threshold_quantile": config.threshold_quantile,
            "feature_config": asdict(config.feature_config),
            "isolation_forest_config": asdict(config.isolation_forest_config),
        },
        "dataset_summary": result.dataset_summary,
        "global_metrics": result.metrics["global"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_max_samples(value: str) -> int | float | Literal["auto"]:
    if value == "auto":
        return "auto"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max-samples must be 'auto', int, or float") from exc


def _parse_contamination(value: str) -> float | Literal["auto"]:
    if value == "auto":
        return "auto"
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("contamination must be 'auto' or float") from exc
    if not 0 < parsed <= 0.5:
        raise argparse.ArgumentTypeError("contamination must be in (0, 0.5]")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
