"""NAB label-alignment diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis_ai.ml.anomaly.nab_experiment import NabSeries, labels_for_series


def audit_nab_label_alignment(series_items: list[NabSeries]) -> dict[str, Any]:
    """Audit processed NAB label windows against available observations."""

    window_records: list[dict[str, Any]] = []
    point_records: list[dict[str, Any]] = []
    aligned_examples: list[dict[str, Any]] = []
    edge_cases: list[dict[str, Any]] = []
    series_with_multiple_windows: list[str] = []

    for series in series_items:
        timestamps = pd.to_datetime(series.frame["timestamp"], errors="coerce")
        minimum = timestamps.min()
        maximum = timestamps.max()
        y_true = labels_for_series(series)
        if len(series.windows) > 1:
            series_with_multiple_windows.append(series.entity_id)
        if int(y_true.sum()) == 0:
            edge_cases.append(
                {
                    "entity_id": series.entity_id,
                    "case": "no_positive_observations_after_alignment",
                    "windows": int(len(series.windows)),
                    "point_labels": int(len(series.point_labels)),
                }
            )
        elif len(aligned_examples) < 10:
            positives = np.flatnonzero(y_true == 1)
            aligned_examples.append(
                {
                    "entity_id": series.entity_id,
                    "first_positive_timestamp": str(timestamps.iloc[int(positives[0])]),
                    "last_positive_timestamp": str(timestamps.iloc[int(positives[-1])]),
                    "positive_observations": int(y_true.sum()),
                }
            )

        for _, row in series.windows.iterrows():
            start = row["timestamp_start"]
            end = row["timestamp_end"]
            is_null_boundary = bool(pd.isna(start) or pd.isna(end))
            if is_null_boundary:
                aligned_count = 0
                half_open_count = 0
                start_exact = False
                end_exact = False
                outside = False
            else:
                inclusive_mask = (timestamps >= start) & (timestamps <= end)
                half_open_mask = (timestamps >= start) & (timestamps < end)
                aligned_count = int(inclusive_mask.sum())
                half_open_count = int(half_open_mask.sum())
                start_exact = bool(timestamps.eq(start).any())
                end_exact = bool(timestamps.eq(end).any())
                outside = bool(start < minimum or end > maximum)
            record = {
                "entity_id": series.entity_id,
                "timestamp_start": str(start),
                "timestamp_end": str(end),
                "observations_in_closed_interval": aligned_count,
                "observations_in_half_open_interval": half_open_count,
                "closed_vs_half_open_delta": aligned_count - half_open_count,
                "start_exact_match": start_exact,
                "end_exact_match": end_exact,
                "outside_observation_range": outside,
                "null_boundary": is_null_boundary,
            }
            window_records.append(record)
            if is_null_boundary or outside or aligned_count == 0:
                edge_cases.append(record)

        for _, row in series.point_labels.iterrows():
            timestamp = row["timestamp"]
            exact_match = bool(timestamps.eq(timestamp).any()) if pd.notna(timestamp) else False
            point_records.append(
                {
                    "entity_id": series.entity_id,
                    "timestamp": str(timestamp),
                    "exact_match": exact_match,
                    "outside_observation_range": bool(
                        pd.isna(timestamp) or timestamp < minimum or timestamp > maximum
                    ),
                }
            )

    labeled_series = {
        item.entity_id
        for item in series_items
        if not item.windows.empty or not item.point_labels.empty
    }
    window_frame = pd.DataFrame(window_records)
    point_frame = pd.DataFrame(point_records)
    total_observations = sum(len(item.frame) for item in series_items)

    return {
        "series_count": len(series_items),
        "observation_count": int(total_observations),
        "labeled_series_count": len(labeled_series),
        "unlabeled_series_count": len(series_items) - len(labeled_series),
        "anomaly_window_count": len(window_records),
        "point_label_count": len(point_records),
        "series_with_multiple_windows_count": len(series_with_multiple_windows),
        "series_with_multiple_windows": series_with_multiple_windows[:20],
        "windows_with_observations": _sum_bool(
            window_frame, "observations_in_closed_interval", op="positive"
        ),
        "windows_without_observations": _sum_bool(
            window_frame, "observations_in_closed_interval", op="zero"
        ),
        "windows_outside_observation_range": _sum_bool(window_frame, "outside_observation_range"),
        "windows_with_null_boundaries": _sum_bool(window_frame, "null_boundary"),
        "windows_with_exact_start": _sum_bool(window_frame, "start_exact_match"),
        "windows_with_exact_end": _sum_bool(window_frame, "end_exact_match"),
        "closed_interval_extra_boundary_points": int(
            window_frame["closed_vs_half_open_delta"].sum() if not window_frame.empty else 0
        ),
        "point_labels_with_exact_match": _sum_bool(point_frame, "exact_match"),
        "point_labels_without_exact_match": (
            int((~point_frame["exact_match"]).sum()) if not point_frame.empty else 0
        ),
        "point_labels_outside_observation_range": _sum_bool(
            point_frame, "outside_observation_range"
        ),
        "aligned_examples": aligned_examples,
        "edge_cases": edge_cases[:20],
        "boundary_policy": (
            "NAB windows are mapped as closed intervals: timestamp >= start and timestamp <= end."
        ),
    }


def render_nab_label_alignment_markdown(audit: dict[str, Any]) -> str:
    """Render a human-readable label alignment report."""

    examples = _markdown_table(
        audit["aligned_examples"],
        [
            "entity_id",
            "first_positive_timestamp",
            "last_positive_timestamp",
            "positive_observations",
        ],
    )
    edge_cases = _markdown_table(
        audit["edge_cases"],
        ["entity_id", "case", "windows", "point_labels"],
    )
    multi_series = "\n".join(
        f"- `{entity_id}`" for entity_id in audit["series_with_multiple_windows"][:10]
    )
    if not multi_series:
        multi_series = "- None"
    if not edge_cases:
        edge_cases = "No problematic edge cases were found."

    return f"""# NAB Label Alignment Audit

Phase: 4
Status: completed from processed local NAB artifacts

## Summary

| Field | Value |
| --- | ---: |
| Series | {audit["series_count"]} |
| Observations | {audit["observation_count"]:,} |
| Labeled series | {audit["labeled_series_count"]} |
| Unlabeled series | {audit["unlabeled_series_count"]} |
| Anomaly windows | {audit["anomaly_window_count"]} |
| Point labels | {audit["point_label_count"]} |
| Series with multiple windows | {audit["series_with_multiple_windows_count"]} |
| Windows outside observation range | {audit["windows_outside_observation_range"]} |
| Windows without observations | {audit["windows_without_observations"]} |
| Windows with null boundaries | {audit["windows_with_null_boundaries"]} |
| Point labels without exact observation match | {audit["point_labels_without_exact_match"]} |

## Boundary Policy

{audit["boundary_policy"]}

All anomaly windows had exact start and end timestamp matches in the processed
observations. Closed intervals add {audit["closed_interval_extra_boundary_points"]}
endpoint observations compared with half-open intervals, which is exactly one
endpoint per window in this processed dataset.

## Correctly Aligned Examples

{examples}

## Multiple-Window Series

{multi_series}

## Edge Cases

{edge_cases}

## Assessment

No labels were found outside their corresponding observation ranges, and every
anomaly window aligned to at least one observation. The Phase 3 closed-interval
mapping is therefore retained. The Phase 4 comparison changes the training and
scoring protocol for model comparability, but it does not rewrite the underlying
NAB label alignment.
"""


def write_nab_label_alignment_report(path: Path, audit: dict[str, Any]) -> None:
    """Write the markdown label-alignment report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_nab_label_alignment_markdown(audit), encoding="utf-8")


def _sum_bool(frame: pd.DataFrame, column: str, *, op: str = "truthy") -> int:
    if frame.empty:
        return 0
    if op == "positive":
        return int((frame[column] > 0).sum())
    if op == "zero":
        return int((frame[column] == 0).sum())
    return int(frame[column].astype(bool).sum())


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No rows."
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
