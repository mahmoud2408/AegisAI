"""Timestamp parsing and ordering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from aegis_ai.data.models.common import QualityFlag

DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S,%f",
    "%Y-%m-%d",
    "%y/%m/%d %H:%M:%S",
    "%y/%m/%d",
    "%Y.%m.%d",
    "%m%d%y %H%M%S",
)


@dataclass(frozen=True)
class ParsedTimestamp:
    """Parsed timestamp plus source-preservation metadata."""

    timestamp: datetime | None
    original: str | None
    timezone_assumption: str | None
    quality_flag: QualityFlag
    warning: str | None = None

    @property
    def timestamp_original(self) -> str | None:
        """Storage-facing alias for the preserved source timestamp."""

        return self.original


def parse_timestamp(
    value: str | int | float | None,
    *,
    timezone: str | None = None,
) -> ParsedTimestamp:
    """Parse a source timestamp without inventing missing timezone information."""

    if value is None:
        return ParsedTimestamp(None, None, None, QualityFlag.MISSING, "missing timestamp")

    original = str(value).strip().strip('"')
    if not original:
        return ParsedTimestamp(None, original, None, QualityFlag.MISSING, "empty timestamp")

    parsed = _parse_datetime(original)
    if parsed is None:
        return ParsedTimestamp(None, original, None, QualityFlag.INVALID, "invalid timestamp")

    if timezone and timezone.upper() == "UTC":
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        parsed = parsed.astimezone(UTC)
        assumption = "source documented or configured as UTC"
    else:
        assumption = "timezone unavailable in source; preserving naive timestamp"

    return ParsedTimestamp(parsed, original, assumption, QualityFlag.VALID)


def combine_date_time(date_value: str | None, time_value: str | None) -> str | None:
    """Combine separate date and time fields from log files."""

    if date_value is None and time_value is None:
        return None
    if date_value is None:
        return str(time_value)
    if time_value is None:
        return str(date_value)
    return f"{date_value} {time_value}"


def flag_temporal_order(
    current: datetime | None,
    previous: datetime | None,
    duplicate: bool = False,
) -> QualityFlag:
    """Return a quality flag for timestamp order checks."""

    if current is None:
        return QualityFlag.INVALID
    if duplicate:
        return QualityFlag.DUPLICATE
    if previous is not None and current < previous:
        return QualityFlag.OUT_OF_ORDER
    return QualityFlag.VALID


def _parse_datetime(value: str) -> datetime | None:
    clean = value.strip().replace(",", ".")
    if clean.isdigit() and len(clean) >= 10:
        try:
            return datetime.fromtimestamp(int(clean[:10]), tz=UTC).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            pass
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue
    return None
