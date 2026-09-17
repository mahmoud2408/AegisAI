"""Deterministic anomaly-detection baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingZScoreBaselineConfig:
    """Configuration for a rolling z-score baseline."""

    score_column: str = "rolling_zscore_long"
    threshold_quantile: float = 0.99

    def __post_init__(self) -> None:
        if not 0 < self.threshold_quantile < 1:
            raise ValueError("threshold_quantile must be between 0 and 1")


class RollingZScoreBaseline:
    """Use absolute causal rolling z-score as an anomaly score."""

    def __init__(self, config: RollingZScoreBaselineConfig | None = None) -> None:
        self.config = config or RollingZScoreBaselineConfig()

    def fit(self, features: pd.DataFrame) -> RollingZScoreBaseline:
        """Keep a model-like interface; this baseline has no learned parameters."""

        if self.config.score_column not in features.columns:
            raise ValueError(f"missing score column: {self.config.score_column}")
        return self

    def score_samples(self, features: pd.DataFrame) -> np.ndarray:
        """Return larger-is-more-anomalous scores."""

        if self.config.score_column not in features.columns:
            raise ValueError(f"missing score column: {self.config.score_column}")
        scores = pd.to_numeric(features[self.config.score_column], errors="coerce").abs()
        return cast(np.ndarray, scores.fillna(0.0).to_numpy(dtype=float))

    def predict(self, features: pd.DataFrame, *, threshold: float) -> np.ndarray:
        """Predict anomalies using an externally selected threshold."""

        return (self.score_samples(features) >= threshold).astype(int)


def quantile_threshold(scores: np.ndarray, quantile: float) -> float:
    """Compute a deterministic threshold from finite validation scores."""

    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    finite_scores = np.asarray(scores, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size == 0:
        raise ValueError("cannot compute threshold from empty scores")
    return float(np.quantile(finite_scores, quantile))
