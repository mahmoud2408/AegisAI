from __future__ import annotations

import argparse

from download_all import main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected LogHub small samples.")
    parser.add_argument(
        "--dataset",
        default="all-small",
        help="One of hdfs, bgl, openstack, hadoop, spark, zookeeper, all-small.",
    )
    parser.add_argument("--large", action="store_true", help="Reserved for future full datasets.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    argv = ["--dataset", args.dataset]
    if args.large:
        argv.append("--large")
    raise SystemExit(main(argv))
