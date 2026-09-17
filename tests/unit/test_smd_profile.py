from __future__ import annotations

import json
from pathlib import Path

from aegis_ai.ml.anomaly.smd_profile import profile_smd_dataset, write_smd_profile_artifacts


def test_profile_smd_dataset_summarizes_raw_structure(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "smd"
    for name in ("train", "test", "test_label", "interpretation_label"):
        (raw_root / name).mkdir(parents=True)
    row = ",".join(str(index / 10) for index in range(38))
    (raw_root / "train" / "machine-1-1.txt").write_text(f"{row}\n{row}\n", encoding="utf-8")
    (raw_root / "test" / "machine-1-1.txt").write_text(f"{row}\n{row}\n{row}\n", encoding="utf-8")
    (raw_root / "test_label" / "machine-1-1.txt").write_text("0\n1\n1\n", encoding="utf-8")
    (raw_root / "interpretation_label" / "machine-1-1.txt").write_text(
        "1-2:0,3,5\n",
        encoding="utf-8",
    )
    lineage = tmp_path / "data" / "manifests" / "lineage"
    lineage.mkdir(parents=True)
    (lineage / "smd.json").write_text(
        json.dumps({"output_records": {"metrics": 190}}),
        encoding="utf-8",
    )

    profile = profile_smd_dataset(tmp_path)

    assert profile["machine_count"] == 1
    assert profile["metric_count"] == 38
    assert profile["train_rows"] == 2
    assert profile["test_rows"] == 3
    assert profile["positive_test_labels"] == 2
    assert profile["anomaly_segments"] == 1
    assert profile["interpretation_intervals"] == 1
    assert profile["processed_metric_records"] == 190
    assert profile["machine_profile"][0]["affected_metric_mentions"] == 3


def test_write_smd_profile_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "profile"
    profile = {
        "dataset_id": "smd",
        "machine_profile": [{"machine_id": "machine-1-1", "train_rows": 2}],
    }

    write_smd_profile_artifacts(profile, output_dir)

    assert (output_dir / "smd_profile.json").exists()
    assert (output_dir / "smd_machine_profile.csv").exists()
