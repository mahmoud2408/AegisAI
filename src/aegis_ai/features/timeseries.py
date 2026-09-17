"""Causal time-series feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalRollingFeatureConfig:
    """Configuration for causal rolling telemetry features."""

    short_window: int = 5
    long_window: int = 20
    trend_window: int = 5
    min_periods: int = 3
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.short_window < 1 or self.long_window < 1 or self.trend_window < 1:
            raise ValueError("rolling windows must be positive")
        if self.min_periods < 1:
            raise ValueError("min_periods must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


FEATURE_COLUMNS = (
    "raw_value",
    "diff_1",
    "pct_change_1",
    "rolling_mean_short",
    "rolling_std_short",
    "rolling_zscore_short",
    "rolling_mean_long",
    "rolling_std_long",
    "rolling_zscore_long",
    "trend_short",
)


def build_causal_rolling_features(
    frame: pd.DataFrame,
    *,
    value_column: str = "metric_value",
    config: CausalRollingFeatureConfig | None = None,
) -> pd.DataFrame:
    """Create features at time t using only value_t and values before t.

    Rolling statistics are computed from ``value.shift(1)`` so the current value does
    not influence its own historical baseline. The current raw value, first
    difference, and percentage change are available at event time and are therefore
    causal.
    """

    if value_column not in frame.columns:
        raise ValueError(f"missing value column: {value_column}")

    cfg = config or CausalRollingFeatureConfig()
    result = frame.copy()
    values = pd.to_numeric(result[value_column], errors="coerce").astype(float)
    history = values.shift(1)

    result["raw_value"] = values
    result["diff_1"] = values - values.shift(1)
    previous = values.shift(1)
    diff = values - previous
    pct_change = diff / previous.where(previous.abs() > cfg.epsilon)
    pct_change = pct_change.mask((previous.abs() <= cfg.epsilon) & (diff.abs() <= cfg.epsilon), 0.0)
    pct_change = pct_change.mask(
        (previous.abs() <= cfg.epsilon) & (diff.abs() > cfg.epsilon),
        diff / cfg.epsilon,
    )
    result["pct_change_1"] = pct_change.replace([np.inf, -np.inf], np.nan)

    _add_rolling_features(
        result,
        values=values,
        history=history,
        window=cfg.short_window,
        suffix="short",
        min_periods=cfg.min_periods,
        epsilon=cfg.epsilon,
    )
    _add_rolling_features(
        result,
        values=values,
        history=history,
        window=cfg.long_window,
        suffix="long",
        min_periods=cfg.min_periods,
        epsilon=cfg.epsilon,
    )
    result["trend_short"] = history.rolling(
        cfg.trend_window,
        min_periods=cfg.min_periods,
    ).apply(_rolling_slope, raw=True)
    return result


def feature_matrix(frame: pd.DataFrame, columns: tuple[str, ...] = FEATURE_COLUMNS) -> np.ndarray:
    """Return a finite numeric feature matrix."""

    return cast(np.ndarray, frame.loc[:, list(columns)].to_numpy(dtype=float))


def finite_feature_mask(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Return rows with complete finite feature values."""

    return np.isfinite(feature_matrix(frame, columns)).all(axis=1)


def _add_rolling_features(
    result: pd.DataFrame,
    *,
    values: pd.Series,
    history: pd.Series,
    window: int,
    suffix: str,
    min_periods: int,
    epsilon: float,
) -> None:
    mean = history.rolling(window, min_periods=min_periods).mean()
    std = history.rolling(window, min_periods=min_periods).std(ddof=0)
    diff = values - mean
    zscore = diff / std.where(std.abs() > epsilon)
    zscore = zscore.mask((std.abs() <= epsilon) & (diff.abs() <= epsilon), 0.0)
    zscore = zscore.mask((std.abs() <= epsilon) & (diff.abs() > epsilon), diff / epsilon)
    result[f"rolling_mean_{suffix}"] = mean
    result[f"rolling_std_{suffix}"] = std
    result[f"rolling_zscore_{suffix}"] = zscore.replace([np.inf, -np.inf], np.nan)


def _rolling_slope(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size < 2:
        return float("nan")
    x = np.arange(clean.size, dtype=float)
    x = x - x.mean()
    denominator = float(np.sum(x * x))
    if denominator == 0:
        return 0.0
    y = clean - clean.mean()
    return float(np.sum(x * y) / denominator)
