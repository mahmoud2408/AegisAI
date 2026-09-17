from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.data.dataset_bootstrap import (  # noqa: E402
    download_many,
    list_available_datasets,
    resolve_dataset_selection,
)
from aegis_ai.data.manifest import directory_size, format_bytes  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download AegisAI public datasets safely.")
    parser.add_argument("--list", action="store_true", help="List configured datasets and exit.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset selector. Examples: nab, smd, smap-msl, loghub, hdfs, bgl, "
            "openstack, all-small, metropt, ai4i, public, all."
        ),
    )
    parser.add_argument("--large", action="store_true", help="Allow datasets marked large/manual.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing downloaded files.")
    parser.add_argument(
        "--retries", type=int, default=3, help="Retry count for transient failures."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root. Defaults to the detected project root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        _print_dataset_list()
        return 0

    selections = args.dataset or ["public"]
    dataset_ids: list[str] = []
    for selection in selections:
        try:
            dataset_ids.extend(resolve_dataset_selection(selection, include_large=args.large))
        except ValueError as exc:
            parser.error(str(exc))

    results = download_many(
        dataset_ids,
        root=args.root,
        include_large=args.large,
        force=args.force,
        retries=args.retries,
        timeout_seconds=args.timeout,
    )
    _print_storage_report(results)
    return 1 if any(result.status == "failed" for result in results) else 0


def _print_dataset_list() -> None:
    print("Configured datasets:")
    for definition in list_available_datasets():
        access = "manual" if definition.requires_manual_access else "public"
        size_policy = "large" if definition.large else "default"
        aliases = f" aliases={','.join(definition.aliases)}" if definition.aliases else ""
        print(
            f"- {definition.id:<20} {access:<8} {size_policy:<8} {definition.destination}{aliases}"
        )


def _print_storage_report(results) -> None:
    print("\nStorage report:")
    for result in results:
        size = format_bytes(directory_size(result.local_path))
        print(f"- {result.dataset_id:<20} {result.status:<16} {size:>10} {result.local_path}")


if __name__ == "__main__":
    raise SystemExit(main())
