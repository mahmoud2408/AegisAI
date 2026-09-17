"""Run the Phase 10 deterministic LogHub intelligence experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.logs import (  # noqa: E402
    load_log_intelligence_config,
    run_log_intelligence_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 10 LogHub intelligence experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/logs/log_intelligence.yaml"),
        help="Path to the log intelligence YAML config.",
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
        help="Optional output root. Defaults to experiments/logs.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_log_intelligence_config(config_path)
    result = run_log_intelligence_experiment(
        root=root,
        output_root=args.output_root,
        run_id=args.run_id,
        config=config,
    )
    payload: dict[str, Any] = {
        "run_dir": str(result.run_dir),
        "datasets_analyzed": result.metrics["datasets_analyzed"],
        "log_evidence_signal_count": result.metrics["log_evidence_signal_count"],
        "log_risk_signal_count": result.metrics["log_risk_signal_count"],
        "label_evaluation": result.metrics["label_evaluation"],
        "elapsed_seconds": result.metrics["elapsed_seconds"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
