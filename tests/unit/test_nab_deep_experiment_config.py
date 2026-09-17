from __future__ import annotations

from pathlib import Path

from aegis_ai.ml.anomaly.nab_deep_experiment import load_nab_deep_config


def test_nab_deep_config_loads_external_yaml() -> None:
    config = load_nab_deep_config(Path("configs/experiments/nab_deep_anomaly.yaml"))

    assert config.train_fraction == 0.5
    assert config.validation_fraction == 0.2
    assert config.windowing.sequence_length == 32
    assert config.threshold.strategy == "train_percentile"
    assert config.dense_autoencoder_config.hidden_dims == (32, 16)
    assert config.ablation.enabled is True
