from __future__ import annotations

import numpy as np

from aegis_ai.ml.anomaly.isolation_forest import (
    IsolationForestConfig,
    IsolationForestDetector,
)


def test_isolation_forest_scores_anomalies_higher_than_normal_points() -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(0, 0.2, size=(128, 3))
    normal = rng.normal(0, 0.2, size=(16, 3))
    anomalies = rng.normal(5, 0.2, size=(4, 3))

    detector = IsolationForestDetector(IsolationForestConfig(n_estimators=50, random_state=7)).fit(
        train
    )

    assert detector.score_samples(anomalies).mean() > detector.score_samples(normal).mean()


def test_isolation_forest_is_reproducible_and_persistable(tmp_path) -> None:
    rng = np.random.default_rng(5)
    train = rng.normal(size=(64, 2))
    test = rng.normal(size=(8, 2))
    config = IsolationForestConfig(n_estimators=25, random_state=11)

    first = IsolationForestDetector(config).fit(train)
    second = IsolationForestDetector(config).fit(train)
    path = first.save(tmp_path / "detector.joblib")
    loaded = IsolationForestDetector.load(path)

    assert np.allclose(first.score_samples(test), second.score_samples(test))
    assert np.allclose(first.score_samples(test), loaded.score_samples(test))
