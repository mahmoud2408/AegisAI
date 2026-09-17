from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.data.validation import format_validation_report, validate_all  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate local AegisAI dataset installation.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root. Defaults to the detected project root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when public datasets are incomplete.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results = validate_all(args.root)
    print(format_validation_report(results))

    if args.strict and any(result.status == "INCOMPLETE" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
