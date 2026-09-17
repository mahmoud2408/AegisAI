from __future__ import annotations

import numpy as np
import pytest

from aegis_ai.features.normalization import IdentityTimeSeriesScaler, TimeSeriesStandardScaler


def test_standard_scaler_uses_training_statistics_only() -> None:
    train = np.asarray([[1.0], [2.0], [3.0]])
    test = np.asarray([[1_000.0]])
    scaler = TimeSeriesStandardScaler().fit(train)

    transformed_train = scaler.transform(train)
    transformed_test = scaler.transform(test)

    assert scaler.mean_.tolist() == [2.0]
    assert pytest.approx(float(transformed_train.mean())) == 0.0
    assert float(transformed_test[0, 0]) > 1_000.0


def test_standard_scaler_handles_constant_features() -> None:
    train = np.asarray([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    scaler = TimeSeriesStandardScaler().fit(train)

    assert scaler.scale_[0] == 1.0
    assert np.isfinite(scaler.transform(train)).all()


def test_identity_scaler_returns_float_copy() -> None:
    values = np.asarray([[1, 2], [3, 4]])
    transformed = IdentityTimeSeriesScaler().fit_transform(values)

    assert transformed.dtype == float
    assert transformed.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert transformed is not values
