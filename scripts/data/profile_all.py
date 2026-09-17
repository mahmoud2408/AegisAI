from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.data.adapters import PRIMARY_DATASETS, available_adapter_ids  # noqa: E402
from aegis_ai.data.preprocessing.pipeline import profile_many  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate adapter-level dataset profiles.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=available_adapter_ids(),
        help="Dataset to profile. Repeat to profile multiple. Defaults to Phase 2 primary set.",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_ids = tuple(args.dataset) if args.dataset else PRIMARY_DATASETS
    profiles = profile_many(dataset_ids, root=args.root)
    print(json.dumps(profiles, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
