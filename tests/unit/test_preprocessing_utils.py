from __future__ import annotations

import pytest

from aegis_ai.data.models import QualityFlag
from aegis_ai.data.preprocessing.splitting import (
    SplitName,
    assert_no_overlapping_windows,
    chronological_split,
    entity_split,
)
from aegis_ai.data.preprocessing.timestamps import combine_date_time, parse_timestamp


def test_timestamp_parser_preserves_naive_timezone_assumption() -> None:
    parsed = parse_timestamp(combine_date_time("2017-05-16", "00:00:00.008"))

    assert parsed.timestamp is not None
    assert parsed.timestamp_original == "2017-05-16 00:00:00.008"
    assert (
        parsed.timezone_assumption == "timezone unavailable in source; preserving naive timestamp"
    )
    assert parsed.quality_flag == QualityFlag.VALID


def test_chronological_split_is_ordered_and_non_random() -> None:
    splits = chronological_split(100, train_fraction=0.7, validation_fraction=0.15)

    assert splits[0].name == SplitName.TRAIN
    assert splits[0].start == 0
    assert splits[0].end == 70
    assert splits[1].start == 70
    assert splits[2].end == 100


def test_entity_split_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        entity_split(["a"], validation_entities={"a"}, test_entities={"a"})


def test_window_overlap_guard_blocks_leakage() -> None:
    with pytest.raises(ValueError):
        assert_no_overlapping_windows(90, 95, window_size=10)
