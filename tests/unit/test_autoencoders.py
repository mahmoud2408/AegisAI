from __future__ import annotations

import numpy as np

from aegis_ai.ml.anomaly.autoencoder import DenseAutoencoderConfig, DenseAutoencoderDetector
from aegis_ai.ml.anomaly.lstm_autoencoder import LSTMAutoencoderConfig, LSTMAutoencoderDetector


def _toy_windows() -> np.ndarray:
    base = np.linspace(0, 1, 48, dtype=np.float32).reshape(24, 2, 1)
    return base


def _toy_multivariate_windows() -> np.ndarray:
    base = np.linspace(0, 1, 180, dtype=np.float32).reshape(30, 3, 2)
    base[:, :, 1] = np.flip(base[:, :, 0], axis=1)
    return base


def test_dense_autoencoder_reconstruction_error_shape() -> None:
    windows = _toy_windows()
    detector = DenseAutoencoderDetector(
        DenseAutoencoderConfig(
            hidden_dims=(4,),
            latent_dim=2,
            epochs=1,
            batch_size=8,
            random_seed=7,
        )
    )

    detector.fit(windows, validation_windows=windows[:8])
    scores = detector.reconstruction_error(windows[:5])

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()
    assert detector.predict(windows[:5], threshold=float(scores.mean())).shape == (5,)


def test_dense_autoencoder_per_feature_reconstruction_error_shape() -> None:
    windows = _toy_multivariate_windows()
    detector = DenseAutoencoderDetector(
        DenseAutoencoderConfig(
            hidden_dims=(8,),
            latent_dim=3,
            epochs=1,
            batch_size=10,
            random_seed=7,
        )
    )

    detector.fit(windows, validation_windows=windows[:8])
    scores = detector.reconstruction_error_by_feature(windows[:5])

    assert scores.shape == (5, 2)
    assert np.isfinite(scores).all()


def test_lstm_autoencoder_reconstruction_error_shape() -> None:
    windows = _toy_windows()
    detector = LSTMAutoencoderDetector(
        LSTMAutoencoderConfig(
            hidden_size=4,
            epochs=1,
            batch_size=8,
            random_seed=7,
        )
    )

    detector.fit(windows, validation_windows=windows[:8])
    scores = detector.reconstruction_error(windows[:5])

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()
    assert detector.predict(windows[:5], threshold=float(scores.mean())).shape == (5,)


def test_lstm_autoencoder_per_feature_reconstruction_error_shape() -> None:
    windows = _toy_multivariate_windows()
    detector = LSTMAutoencoderDetector(
        LSTMAutoencoderConfig(
            hidden_size=4,
            epochs=1,
            batch_size=10,
            random_seed=7,
        )
    )

    detector.fit(windows, validation_windows=windows[:8])
    scores = detector.reconstruction_error_by_feature(windows[:5])

    assert scores.shape == (5, 2)
    assert np.isfinite(scores).all()


def test_dense_autoencoder_is_reproducible_with_fixed_seed() -> None:
    windows = _toy_windows()
    config = DenseAutoencoderConfig(
        hidden_dims=(4,),
        latent_dim=2,
        epochs=1,
        batch_size=8,
        random_seed=11,
    )

    first = DenseAutoencoderDetector(config)
    second = DenseAutoencoderDetector(config)
    first.fit(windows, validation_windows=windows[:8])
    second.fit(windows, validation_windows=windows[:8])

    np.testing.assert_allclose(
        first.reconstruction_error(windows[:6]),
        second.reconstruction_error(windows[:6]),
        rtol=1e-6,
        atol=1e-6,
    )
