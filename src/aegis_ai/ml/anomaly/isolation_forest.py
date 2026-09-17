"""Isolation Forest detector wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class IsolationForestConfig:
    """Configurable, deterministic Isolation Forest parameters."""

    n_estimators: int = 200
    max_samples: int | float | Literal["auto"] = "auto"
    contamination: float | Literal["auto"] = "auto"
    random_state: int = 42
    n_jobs: int | None = -1

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if self.contamination != "auto" and not 0 < float(self.contamination) <= 0.5:
            raise ValueError("contamination must be 'auto' or in (0, 0.5]")


class IsolationForestDetector:
    """Small stable interface around scikit-learn IsolationForest."""

    def __init__(self, config: IsolationForestConfig | None = None) -> None:
        self.config = config or IsolationForestConfig()
        self.model = IsolationForest(**asdict(self.config))

    def fit(self, features: np.ndarray) -> Self:
        """Fit the detector on historical feature rows."""

        self.model.fit(_as_matrix(features))
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Return larger-is-more-anomalous scores."""

        return cast(np.ndarray, -self.model.score_samples(_as_matrix(features)))

    def predict(self, features: np.ndarray, *, threshold: float) -> np.ndarray:
        """Predict anomalies using an externally selected threshold."""

        return (self.score_samples(features) >= threshold).astype(int)

    def save(self, path: Path) -> Path:
        """Persist detector configuration and fitted model."""

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": asdict(self.config), "model": self.model}, path)
        return path

    @classmethod
    def load(cls, path: Path) -> IsolationForestDetector:
        """Load a detector saved by :meth:`save`."""

        payload: dict[str, Any] = joblib.load(path)
        detector = cls(IsolationForestConfig(**payload["config"]))
        detector.model = payload["model"]
        return detector


def _as_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if matrix.shape[0] == 0:
        raise ValueError("features must contain at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError("features must be finite")
    return matrix
