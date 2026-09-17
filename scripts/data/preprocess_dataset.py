from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.data.adapters import available_adapter_ids  # noqa: E402
from aegis_ai.data.preprocessing.pipeline import preprocess_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess one AegisAI dataset.")
    parser.add_argument("--dataset", required=True, choices=available_adapter_ids())
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=50_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = preprocess_dataset(args.dataset, root=args.root, batch_size=args.batch_size)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.status != "INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
