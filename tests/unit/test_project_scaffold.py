from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


REQUIRED_DOCS = [
    "docs/architecture.md",
    "docs/datasets.md",
    "docs/methodology.md",
    "docs/experiments.md",
    "docs/model-card.md",
    "docs/api.md",
    "docs/rag.md",
    "docs/agent.md",
    "docs/deployment.md",
    "docs/research.md",
]


def test_required_documentation_files_exist() -> None:
    missing = [path for path in REQUIRED_DOCS if not (ROOT / path).is_file()]

    assert missing == []


def test_raw_dataset_directory_does_not_contain_committed_data() -> None:
    raw_dir = ROOT / "data" / "raw"
    allowed_names = {".gitkeep"}
    committed_payloads = [
        path.name for path in raw_dir.iterdir() if path.is_file() and path.name not in allowed_names
    ]

    assert committed_payloads == []


def test_source_package_boundaries_exist() -> None:
    packages = [
        "api",
        "agents",
        "data",
        "db",
        "decision",
        "domain",
        "features",
        "forecasting",
        "llm",
        "ml",
        "observability",
        "rag",
        "rca",
        "shared",
    ]

    missing = [
        package
        for package in packages
        if not (ROOT / "src" / "aegis_ai" / package / "__init__.py").is_file()
    ]

    assert missing == []
