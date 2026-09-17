"""Leakage-safe normalization utilities for time-series experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass
class TimeSeriesStandardScaler:
    """Standardize features using statistics learned from training data only."""

    epsilon: float = 1e-8
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> TimeSeriesStandardScaler:
        """Fit per-feature mean and standard deviation."""

        array = _as_2d_float_array(values)
        if array.shape[0] == 0:
            raise ValueError("cannot fit scaler on an empty array")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.mean_ = cast(np.ndarray, np.mean(array, axis=0))
        scale = cast(np.ndarray, np.std(array, axis=0, ddof=0))
        self.scale_ = np.where(np.abs(scale) <= self.epsilon, 1.0, scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Transform values using previously fitted training statistics."""

        if self.mean_ is None or self.scale_ is None:
            raise ValueError("scaler must be fitted before transform")
        array = _as_2d_float_array(values)
        return cast(np.ndarray, (array - self.mean_) / self.scale_)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        """Fit and transform the same training array."""

        return self.fit(values).transform(values)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Recover original feature scale."""

        if self.mean_ is None or self.scale_ is None:
            raise ValueError("scaler must be fitted before inverse_transform")
        array = _as_2d_float_array(values)
        return cast(np.ndarray, array * self.scale_ + self.mean_)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scaler parameters for experiment artifacts."""

        return {
            "epsilon": self.epsilon,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TimeSeriesStandardScaler:
        """Rehydrate a scaler from serialized parameters."""

        scaler = cls(epsilon=float(payload["epsilon"]))
        mean = payload.get("mean")
        scale = payload.get("scale")
        scaler.mean_ = np.asarray(mean, dtype=float) if mean is not None else None
        scaler.scale_ = np.asarray(scale, dtype=float) if scale is not None else None
        return scaler


class IdentityTimeSeriesScaler:
    """Scaler interface implementation for intentionally unnormalized ablations."""

    def fit(self, values: np.ndarray) -> IdentityTimeSeriesScaler:
        _as_2d_float_array(values)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return _as_2d_float_array(values).copy()

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": "identity"}


def _as_2d_float_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("expected a 2D array shaped [observations, features]")
    if not np.isfinite(array).all():
        raise ValueError("normalization input contains NaN or infinite values")
    return array
