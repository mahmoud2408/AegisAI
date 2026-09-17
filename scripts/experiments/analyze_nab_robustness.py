"""CLI entrypoint for Phase 5 NAB robustness analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.ml.anomaly.nab_robustness import (  # noqa: E402
    NabRobustnessConfig,
    run_nab_robustness_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 5 NAB robustness diagnostics.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root.")
    parser.add_argument(
        "--phase4-run-dir",
        type=Path,
        default=ROOT
        / "experiments"
        / "anomaly"
        / "nab"
        / "deep_autoencoders"
        / "run_phase4_deep_anomaly",
        help="Existing Phase 4 run directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "nab_deep_anomaly.yaml",
        help="Phase 4 experiment config used for split/model reconstruction.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "anomaly" / "nab" / "robustness",
        help="NAB robustness output root.",
    )
    parser.add_argument(
        "--smd-output-root",
        type=Path,
        default=ROOT / "experiments" / "smd" / "profile",
        help="SMD profile output root.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional NAB run id.")
    parser.add_argument("--smd-run-id", type=str, default=None, help="Optional SMD profile run id.")
    parser.add_argument(
        "--window-sensitivity-series",
        type=int,
        default=6,
        help="Number of labeled NAB series for neural window sensitivity.",
    )
    parser.add_argument(
        "--window-sensitivity-epochs",
        type=int,
        default=2,
        help="Diagnostic epochs per neural window-size model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = NabRobustnessConfig(
        window_sensitivity_series=args.window_sensitivity_series,
        window_sensitivity_epochs=args.window_sensitivity_epochs,
    )
    result = run_nab_robustness_analysis(
        root=args.root,
        phase4_run_dir=args.phase4_run_dir,
        output_root=args.output_root,
        smd_output_root=args.smd_output_root,
        run_id=args.run_id,
        smd_run_id=args.smd_run_id,
        config_path=args.config,
        config=config,
    )
    payload: dict[str, Any] = {
        "run_dir": str(result.run_dir),
        "smd_profile_dir": str(result.smd_profile_dir),
        "series_classification_counts": result.summary["series_classification_counts"],
        "threshold_best_by_f1": result.summary["threshold_best_by_f1"],
        "window_sensitivity_best_by_f1": result.summary["window_sensitivity_best_by_f1"],
        "official_nab_scoring": result.summary["official_nab_scoring"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
