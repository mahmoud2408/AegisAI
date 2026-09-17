from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pyarrow.parquet as pq

from aegis_ai.data.adapters import create_adapter
from aegis_ai.data.preprocessing.pipeline import preprocess_dataset


def test_ai4i_adapter_and_pipeline_on_small_fixture(tmp_path: Path) -> None:
    write_catalog(
        tmp_path,
        """
        datasets:
          - id: ai4i
            name: AI4I Fixture
            category: predictive_maintenance_classification
            description: fixture
            official_source: fixture
            download_method: fixture
            destination: data/raw/ai4i
            expected_files: [ai4i2020.csv]
            requires_manual_access: false
            license: fixture
            notes: fixture
        """,
    )
    raw_dir = tmp_path / "data" / "raw" / "ai4i"
    raw_dir.mkdir(parents=True)
    with (raw_dir / "ai4i2020.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "UDI",
                "Product ID",
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "Machine failure",
                "TWF",
                "HDF",
                "PWF",
                "OSF",
                "RNF",
            ]
        )
        writer.writerow([1, "M1", "M", 298.1, 308.6, 1551, 42.8, 0, 0, 0, 0, 0, 0, 0])
        writer.writerow([2, "M2", "L", 300.1, 310.6, 1400, 50.0, 10, 1, 0, 1, 0, 0, 0])

    adapter = create_adapter("ai4i", tmp_path)
    raw = adapter.load_raw()

    assert adapter.validate(raw).status == "READY"
    assert adapter.profile(raw).row_count == 2
    assert sum(1 for _ in adapter.iter_metric_records(raw)) == 10
    assert sum(1 for _ in adapter.iter_failure_records(raw)) == 2
    assert sum(1 for _ in adapter.get_labels(raw)) == 12

    result = preprocess_dataset("ai4i", root=tmp_path, batch_size=4)

    assert result.output_records["metrics"] == 10
    metric_file = tmp_path / result.output_files["metrics"][0]
    assert pq.ParquetFile(metric_file).metadata.num_rows == 10


def test_loghub_openstack_adapter_extracts_request_id(tmp_path: Path) -> None:
    write_catalog(
        tmp_path,
        """
        datasets:
          - id: loghub-openstack
            name: OpenStack Fixture
            category: log_analytics
            description: fixture
            official_source: fixture
            download_method: fixture
            destination: data/raw/loghub/openstack
            expected_files: [OpenStack_2k.log_structured.csv]
            requires_manual_access: false
            license: fixture
            notes: fixture
        """,
    )
    raw_dir = tmp_path / "data" / "raw" / "loghub" / "openstack"
    raw_dir.mkdir(parents=True)
    with (raw_dir / "OpenStack_2k.log_structured.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "LineId",
                "Logrecord",
                "Date",
                "Time",
                "Pid",
                "Level",
                "Component",
                "ADDR",
                "Content",
                "EventId",
                "EventTemplate",
            ]
        )
        writer.writerow(
            [
                1,
                "nova-api.log",
                "2017-05-16",
                "00:00:00.008",
                "25746",
                "INFO",
                "nova.api",
                "req-38101a0b-2096-447d-96ea-a692162415ae",
                "GET /servers status: 200",
                "E25",
                '<*> "GET <*>" status: <*>',
            ]
        )
    (raw_dir / "OpenStack_2k.log_templates.csv").write_text(
        "EventId,EventTemplate\nE25,template\n",
        encoding="utf-8",
    )

    adapter = create_adapter("loghub-openstack", tmp_path)
    raw = adapter.load_raw()
    event = next(iter(adapter.iter_log_records(raw)))

    assert event["request_id"] == "req-38101a0b-2096-447d-96ea-a692162415ae"
    assert event["event_type"] == "E25"
    assert event["service_id"] == "nova.api"


def write_catalog(tmp_path: Path, text: str) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "datasets.yaml").write_text(textwrap.dedent(text).strip(), encoding="utf-8")
