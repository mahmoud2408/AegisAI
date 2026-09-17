from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.data.adapters import PRIMARY_DATASETS, available_adapter_ids  # noqa: E402
from aegis_ai.data.preprocessing.pipeline import preprocess_many  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess selected AegisAI datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=available_adapter_ids(),
        help="Dataset to preprocess. Repeat to process multiple. Defaults to Phase 2 primary set.",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=50_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_ids = tuple(args.dataset) if args.dataset else PRIMARY_DATASETS
    results = preprocess_many(dataset_ids, root=args.root, batch_size=args.batch_size)
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
    return 0 if all(result.status != "INCOMPLETE" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
