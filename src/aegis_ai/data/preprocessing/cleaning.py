"""Small deterministic cleaning helpers for data adapters."""

from __future__ import annotations

import re
from collections.abc import Iterable

REQUEST_ID_PATTERN = re.compile(r"\breq-[0-9a-fA-F-]{8,}\b")


def blank_to_none(value: object) -> str | None:
    """Normalize blank strings to None while preserving non-empty text."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_float(value: str | None) -> float | None:
    """Parse a float value, returning None for missing or invalid input."""

    if value is None:
        return None
    clean = str(value).strip()
    if not clean:
        return None
    try:
        return float(clean)
    except ValueError:
        return None


def parse_int_label(value: str | None) -> int | None:
    """Parse a binary label without assuming semantics."""

    if value is None:
        return None
    clean = str(value).strip()
    if clean not in {"0", "1"}:
        return None
    return int(clean)


def extract_request_id(*values: str | None) -> str | None:
    """Extract an explicit OpenStack-style request id if one is present."""

    for value in values:
        if not value:
            continue
        match = REQUEST_ID_PATTERN.search(value)
        if match:
            return match.group(0)
    return None


def is_binary_state_column(values: Iterable[str], max_scan: int = 10_000) -> bool:
    """Return True when observed non-empty values are all 0/1-like."""

    observed = 0
    for value in values:
        if observed >= max_scan:
            break
        clean = str(value).strip().lower()
        if not clean:
            continue
        observed += 1
        try:
            numeric = float(clean)
        except ValueError:
            if clean not in {"true", "false"}:
                return False
        else:
            if numeric not in {0.0, 1.0}:
                return False
    return observed > 0


def sanitize_partition_value(value: str | int | None) -> str:
    """Make a value safe for use in a partitioned output path."""

    text = "unknown" if value is None else str(value)
    text = text.replace("\\", "_").replace("/", "_").replace(":", "_")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
