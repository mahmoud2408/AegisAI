"""Leakage-aware split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

T = TypeVar("T")


class SplitName(StrEnum):
    """Supported split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitBounds:
    """Half-open index bounds for a split."""

    name: SplitName
    start: int
    end: int


def chronological_split(
    length: int,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> list[SplitBounds]:
    """Return deterministic chronological half-open split bounds."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave a test split")

    train_end = int(length * train_fraction)
    validation_end = train_end + int(length * validation_fraction)
    return [
        SplitBounds(SplitName.TRAIN, 0, train_end),
        SplitBounds(SplitName.VALIDATION, train_end, validation_end),
        SplitBounds(SplitName.TEST, validation_end, length),
    ]


def entity_split(
    entity_ids: list[str],
    *,
    validation_entities: set[str] | None = None,
    test_entities: set[str] | None = None,
) -> dict[str, SplitName]:
    """Assign entities to non-overlapping splits."""

    validation_entities = validation_entities or set()
    test_entities = test_entities or set()
    overlap = validation_entities.intersection(test_entities)
    if overlap:
        raise ValueError(f"entities cannot be in validation and test: {sorted(overlap)}")

    assignments: dict[str, SplitName] = {}
    for entity_id in entity_ids:
        if entity_id in test_entities:
            assignments[entity_id] = SplitName.TEST
        elif entity_id in validation_entities:
            assignments[entity_id] = SplitName.VALIDATION
        else:
            assignments[entity_id] = SplitName.TRAIN
    return assignments


def assert_no_overlapping_windows(
    train_end: int,
    validation_start: int,
    *,
    window_size: int,
) -> None:
    """Fail when sliding windows would cross a split boundary."""

    if window_size < 1:
        raise ValueError("window_size must be positive")
    if train_end + window_size > validation_start:
        raise ValueError("window overlap would leak train context into validation/test labels")
