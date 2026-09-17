"""Canonical record validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ValidationError


def validate_record_batch(
    records: Iterable[dict[str, Any]],
    model_type: type[BaseModel],
    *,
    max_errors: int = 25,
) -> tuple[int, list[str]]:
    """Validate a batch of dictionaries against a Pydantic canonical model."""

    valid = 0
    errors: list[str] = []
    for index, record in enumerate(records):
        try:
            model_type.model_validate(record)
        except ValidationError as exc:
            if len(errors) < max_errors:
                errors.append(f"record {index}: {exc.errors(include_url=False)}")
        else:
            valid += 1
    return valid, errors
