"""Run Phase 11 document ingestion and chunking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.rag import ingest_knowledge, load_rag_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Phase 11 knowledge documents.")
    parser.add_argument("--config", type=Path, default=Path("configs/rag/rag.yaml"))
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else args.root / args.config
    config = load_rag_config(config_path)
    documents, chunks, stats = ingest_knowledge(root=args.root, config=config)
    payload: dict[str, Any] = {
        "run_dir": str(config.run_dir(args.root)),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "documents_by_domain": stats["documents_by_domain"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
