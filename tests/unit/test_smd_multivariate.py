from __future__ import annotations

from pathlib import Path

import numpy as np

from aegis_ai.ml.anomaly.autoencoder import DenseAutoencoderConfig
from aegis_ai.ml.anomaly.isolation_forest import IsolationForestConfig
from aegis_ai.ml.anomaly.lstm_autoencoder import LSTMAutoencoderConfig
from aegis_ai.ml.anomaly.smd_multivariate import (
    SMDMultivariateExperimentConfig,
    SMDScoreBundle,
    SMDWindowSensitivityConfig,
    evaluate_score_bundle,
    load_smd_machines,
    prepare_smd_machine,
    run_smd_multivariate_experiment,
    score_per_feature_zscore,
    threshold_from_scores,
)


def test_load_smd_machines_preserves_identity_and_dimensions(tmp_path: Path) -> None:
    root = _write_smd_fixture(tmp_path)

    machines = load_smd_machines(root, metric_count=3)

    assert [machine.machine_id for machine in machines] == ["machine-1-1"]
    machine = machines[0]
    assert machine.train.shape == (30, 3)
    assert machine.test.shape == (24, 3)
    assert machine.test_labels.shape == (24,)
    assert machine.metric_names == ("metric_00", "metric_01", "metric_02")
    assert machine.interpretation_intervals[0].affected_metric_indices == (0, 2)


def test_prepare_smd_machine_uses_train_fit_scaler_without_test_leakage(tmp_path: Path) -> None:
    root = _write_smd_fixture(tmp_path, test_shift=100.0)
    machine = load_smd_machines(root, metric_count=3)[0]
    config = _small_config(window_size=4, stride=2)

    prepared = prepare_smd_machine(machine, config)

    np.testing.assert_allclose(prepared.train_fit_scaled.mean(axis=0), 0.0, atol=1e-6)
    assert not np.allclose(prepared.test_scaled.mean(axis=0), 0.0, atol=1e-2)
    assert prepared.train_windows.start_indices.min() == 0
    assert prepared.train_windows.end_indices.max() < prepared.train_fit.shape[0]
    assert prepared.validation_windows.end_indices.max() < prepared.validation.shape[0]


def test_prepare_smd_machine_evaluates_endpoint_labels(tmp_path: Path) -> None:
    labels = np.zeros(24, dtype=int)
    labels[1] = 1
    root = _write_smd_fixture(tmp_path, labels=labels)
    machine = load_smd_machines(root, metric_count=3)[0]
    config = _small_config(window_size=4, stride=1)

    prepared = prepare_smd_machine(machine, config)

    assert prepared.test_eval_indices[0] == 3
    assert prepared.test_windows.labels is not None
    assert prepared.test_windows.labels[0] == 1
    assert prepared.test_eval_labels[0] == 0


def test_threshold_from_scores_uses_requested_split() -> None:
    train_scores = np.asarray([1.0, 2.0, 3.0])
    validation_scores = np.asarray([10.0, 20.0, 30.0])

    train_threshold = threshold_from_scores(
        train_scores=train_scores,
        validation_scores=validation_scores,
        strategy="train_percentile",
        percentile=50,
    )
    validation_threshold = threshold_from_scores(
        train_scores=train_scores,
        validation_scores=validation_scores,
        strategy="validation_percentile",
        percentile=50,
    )

    assert train_threshold == 2.0
    assert validation_threshold == 20.0


def test_zscore_scores_have_global_and_per_metric_outputs(tmp_path: Path) -> None:
    root = _write_smd_fixture(tmp_path)
    machine = load_smd_machines(root, metric_count=3)[0]
    prepared = prepare_smd_machine(machine, _small_config(window_size=4, stride=1))

    scores = score_per_feature_zscore(prepared)

    assert scores.test_scores.shape == prepared.test_eval_labels.shape
    assert scores.per_metric_scores is not None
    assert scores.per_metric_scores.shape == (prepared.test_eval_labels.shape[0], 3)


def test_evaluate_score_bundle_uses_train_threshold_and_scores_predictions(
    tmp_path: Path,
) -> None:
    root = _write_smd_fixture(tmp_path)
    machine = load_smd_machines(root, metric_count=3)[0]
    prepared = prepare_smd_machine(machine, _small_config(window_size=4, stride=1))
    test_scores = np.linspace(0.0, 1.0, prepared.test_eval_labels.shape[0])
    bundle = SMDScoreBundle(
        model_name="per_feature_zscore",
        train_scores=np.asarray([0.1, 0.2, 0.3]),
        validation_scores=np.asarray([0.8, 0.9, 1.0]),
        test_scores=test_scores,
        per_metric_scores=np.repeat(test_scores[:, None], 3, axis=1),
        training_seconds=0.0,
        inference_seconds=0.0,
        parameter_count=0,
        peak_rss_mb=None,
        train_rows=3,
        validation_rows=3,
        test_rows=prepared.test_eval_labels.shape[0],
    )

    outcome = evaluate_score_bundle(
        prepared,
        bundle,
        strategy="train_percentile",
        threshold_percentile=50,
    )

    assert outcome.threshold == 0.2
    assert outcome.predictions.shape == prepared.test_eval_labels.shape
    assert outcome.metrics.support == prepared.test_eval_labels.shape[0]


def test_mini_smd_experiment_writes_core_artifacts(tmp_path: Path) -> None:
    root = _write_smd_fixture(tmp_path)
    output_root = tmp_path / "experiments"
    config = _small_config(window_size=4, stride=2)

    result = run_smd_multivariate_experiment(
        root=root,
        output_root=output_root,
        run_id="unit",
        config=config,
    )

    assert result.dataset_audit["machine_count"] == 1
    assert (result.run_dir / "global_metrics.csv").exists()
    assert (result.run_dir / "per_machine_metrics.csv").exists()
    assert (result.run_dir / "metrics.json").exists()
    assert (result.run_dir / "per_metric_scores" / "dense_autoencoder" / "machine-1-1.npz").exists()


def _small_config(*, window_size: int, stride: int) -> SMDMultivariateExperimentConfig:
    return SMDMultivariateExperimentConfig(
        train_fraction=0.6,
        window_size=window_size,
        stride=stride,
        metric_count=3,
        threshold_percentile=80.0,
        threshold_strategies=("train_percentile",),
        train_row_limit=12,
        train_window_limit=8,
        max_visualized_machines=0,
        isolation_forest_config=IsolationForestConfig(n_estimators=5, random_state=7),
        dense_autoencoder_config=DenseAutoencoderConfig(
            hidden_dims=(6,),
            latent_dim=2,
            batch_size=4,
            epochs=1,
            patience=1,
            random_seed=7,
            device="cpu",
        ),
        lstm_autoencoder_config=LSTMAutoencoderConfig(
            hidden_size=4,
            batch_size=4,
            epochs=1,
            patience=1,
            random_seed=7,
            device="cpu",
        ),
        window_sensitivity=_disabled_sensitivity(),
    )


def _disabled_sensitivity() -> SMDWindowSensitivityConfig:
    return SMDWindowSensitivityConfig(enabled=False, max_machines=1, window_sizes=(4,), epochs=1)


def _write_smd_fixture(
    root: Path,
    *,
    labels: np.ndarray | None = None,
    test_shift: float = 0.0,
) -> Path:
    raw = root / "data" / "raw" / "smd"
    for subdir in ("train", "test", "test_label", "interpretation_label"):
        (raw / subdir).mkdir(parents=True, exist_ok=True)
    train_base = np.arange(90, dtype=np.float32).reshape(30, 3) / 10.0
    test_base = np.arange(72, dtype=np.float32).reshape(24, 3) / 10.0 + test_shift
    label_array = np.zeros(24, dtype=int) if labels is None else labels.astype(int)
    if labels is None:
        label_array[10:13] = 1

    np.savetxt(raw / "train" / "machine-1-1.txt", train_base, delimiter=",", fmt="%.6f")
    np.savetxt(raw / "test" / "machine-1-1.txt", test_base, delimiter=",", fmt="%.6f")
    (raw / "test_label" / "machine-1-1.txt").write_text(
        "\n".join(str(int(value)) for value in label_array) + "\n",
        encoding="utf-8",
    )
    (raw / "interpretation_label" / "machine-1-1.txt").write_text(
        "10-12:0,2\n",
        encoding="utf-8",
    )
    return root
