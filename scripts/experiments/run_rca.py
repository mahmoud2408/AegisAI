"""Run the Phase 9 deterministic incident grouping and RCA experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.rca import load_rca_scoring_config, run_rca_experiment  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 9 deterministic RCA experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rca/rca_scoring.yaml"),
        help="Path to the RCA scoring YAML config.",
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
        help="Optional output root. Defaults to experiments/rca.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_rca_scoring_config(config_path)
    result = run_rca_experiment(
        root=root,
        output_root=args.output_root,
        run_id=args.run_id,
        config=config,
    )
    payload: dict[str, Any] = {
        "run_dir": str(result.run_dir),
        "preliminary_incident_windows": result.metrics["preliminary_incident_windows"],
        "merged_incident_windows": result.metrics["merged_incident_windows"],
        "average_evidence_count_per_incident": result.metrics[
            "average_evidence_count_per_incident"
        ],
        "candidate_count": result.metrics["candidate_count"],
        "elapsed_seconds": result.metrics["elapsed_seconds"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
