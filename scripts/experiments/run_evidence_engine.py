"""Run the Phase 8B evidence engine and risk-signal aggregation experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.evidence import (  # noqa: E402
    load_evidence_scoring_config,
    run_evidence_engine_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 8B evidence engine experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/risk/evidence_scoring.yaml"),
        help="Path to the evidence-scoring YAML config.",
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
        help="Optional output root. Defaults to experiments/evidence.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_evidence_scoring_config(config_path)
    result = run_evidence_engine_experiment(
        root=root,
        output_root=args.output_root,
        run_id=args.run_id,
        config=config,
    )
    payload: dict[str, Any] = {
        "run_dir": str(result.run_dir),
        "evidence_signal_count": result.metrics["evidence_signal_count"],
        "risk_signal_count": result.metrics["risk_signal_count"],
        "mean_risk_score": result.metrics["mean_risk_score"],
        "max_risk_score": result.metrics["max_risk_score"],
        "leakage_detected": result.metrics["leakage_audit"]["leakage_detected"],
        "elapsed_seconds": result.metrics["elapsed_seconds"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
