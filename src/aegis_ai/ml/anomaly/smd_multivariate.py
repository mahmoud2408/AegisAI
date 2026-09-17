"""Phase 6 multivariate anomaly detection experiment on SMD."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes

from aegis_ai.data.adapters.smd import parse_interpretation_line, read_non_empty_lines
from aegis_ai.data.dataset_registry import project_root
from aegis_ai.evaluation.anomaly import (
    BinaryAnomalyMetrics,
    DetectionDelaySummary,
    compute_binary_metrics,
    positive_segments,
)
from aegis_ai.features.normalization import TimeSeriesStandardScaler
from aegis_ai.features.windowing import (
    SequenceWindowConfig,
    SequenceWindowSet,
    make_sequence_windows,
)
from aegis_ai.ml.anomaly.autoencoder import DenseAutoencoderConfig, DenseAutoencoderDetector
from aegis_ai.ml.anomaly.baselines import quantile_threshold
from aegis_ai.ml.anomaly.isolation_forest import IsolationForestConfig, IsolationForestDetector
from aegis_ai.ml.anomaly.lstm_autoencoder import LSTMAutoencoderConfig, LSTMAutoencoderDetector
from aegis_ai.ml.anomaly.smd_profile import SMDProfileConfig, profile_smd_dataset
from aegis_ai.ml.anomaly.torch_training import process_rss_mb, set_reproducible_seed

matplotlib.rcParams.update(
    {
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "font.size": 9,
    }
)

SMDModelName = Literal[
    "per_feature_zscore",
    "isolation_forest",
    "dense_autoencoder",
    "lstm_autoencoder",
]
ThresholdStrategy = Literal["train_percentile", "validation_percentile"]

MODEL_NAMES: tuple[SMDModelName, ...] = (
    "per_feature_zscore",
    "isolation_forest",
    "dense_autoencoder",
    "lstm_autoencoder",
)
THRESHOLD_STRATEGIES: tuple[ThresholdStrategy, ...] = (
    "train_percentile",
    "validation_percentile",
)


@dataclass(frozen=True)
class SMDInterpretationInterval:
    """SMD interpretation interval with affected metric indices."""

    start_index: int
    end_index: int
    affected_metric_indices: tuple[int, ...]


@dataclass(frozen=True)
class SMDMachineData:
    """Wide SMD matrices and labels for one machine."""

    machine_id: str
    train: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    interpretation_intervals: tuple[SMDInterpretationInterval, ...]
    metric_names: tuple[str, ...]


@dataclass(frozen=True)
class SMDWindowSensitivityConfig:
    """Small sequence-window diagnostic for neural SMD models."""

    enabled: bool = True
    max_machines: int = 2
    window_sizes: tuple[int, ...] = (32, 64)
    epochs: int = 1

    def __post_init__(self) -> None:
        if self.max_machines < 1:
            raise ValueError("window_sensitivity.max_machines must be positive")
        if not self.window_sizes or any(size < 2 for size in self.window_sizes):
            raise ValueError("window_sensitivity.window_sizes must contain values >= 2")
        if self.epochs < 1:
            raise ValueError("window_sensitivity.epochs must be positive")


@dataclass(frozen=True)
class SMDMultivariateExperimentConfig:
    """Configuration for the SMD multivariate anomaly experiment."""

    train_fraction: float = 0.8
    window_size: int = 64
    stride: int = 1
    metric_count: int = 38
    scaler_epsilon: float = 1e-3
    threshold_percentile: float = 99.0
    threshold_strategies: tuple[ThresholdStrategy, ...] = THRESHOLD_STRATEGIES
    max_machines: int | None = None
    train_row_limit: int | None = 12000
    train_window_limit: int | None = 4096
    top_k_metrics: int = 5
    max_top_metric_rows_per_outcome: int = 200
    max_visualized_machines: int = 6
    random_seed: int = 42
    isolation_forest_config: IsolationForestConfig = field(
        default_factory=lambda: IsolationForestConfig(n_estimators=100, random_state=42)
    )
    dense_autoencoder_config: DenseAutoencoderConfig = field(
        default_factory=lambda: DenseAutoencoderConfig(
            hidden_dims=(128, 64),
            latent_dim=16,
            learning_rate=1e-3,
            batch_size=256,
            epochs=1,
            patience=1,
            random_seed=42,
            device="cpu",
        )
    )
    lstm_autoencoder_config: LSTMAutoencoderConfig = field(
        default_factory=lambda: LSTMAutoencoderConfig(
            hidden_size=32,
            num_layers=1,
            learning_rate=1e-3,
            batch_size=256,
            epochs=1,
            patience=1,
            random_seed=42,
            device="cpu",
        )
    )
    window_sensitivity: SMDWindowSensitivityConfig = field(
        default_factory=SMDWindowSensitivityConfig
    )

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.metric_count < 1:
            raise ValueError("metric_count must be positive")
        if self.scaler_epsilon <= 0:
            raise ValueError("scaler_epsilon must be positive")
        if not 0 < self.threshold_percentile < 100:
            raise ValueError("threshold_percentile must be in (0, 100)")
        if not self.threshold_strategies:
            raise ValueError("threshold_strategies must not be empty")
        invalid = set(self.threshold_strategies).difference(THRESHOLD_STRATEGIES)
        if invalid:
            raise ValueError(f"unsupported threshold strategies: {sorted(invalid)}")
        if self.max_machines is not None and self.max_machines < 1:
            raise ValueError("max_machines must be positive when provided")
        if self.train_row_limit is not None and self.train_row_limit < 1:
            raise ValueError("train_row_limit must be positive when provided")
        if self.train_window_limit is not None and self.train_window_limit < 1:
            raise ValueError("train_window_limit must be positive when provided")
        if self.top_k_metrics < 1:
            raise ValueError("top_k_metrics must be positive")
        if self.max_top_metric_rows_per_outcome < 1:
            raise ValueError("max_top_metric_rows_per_outcome must be positive")
        if self.max_visualized_machines < 0:
            raise ValueError("max_visualized_machines must be non-negative")


@dataclass(frozen=True)
class SMDPreparedMachine:
    """Leakage-safe arrays and windows for one SMD machine."""

    machine: SMDMachineData
    train_fit: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_fit_scaled: np.ndarray
    validation_scaled: np.ndarray
    test_scaled: np.ndarray
    train_windows: SequenceWindowSet
    validation_windows: SequenceWindowSet
    test_windows: SequenceWindowSet
    test_eval_indices: np.ndarray
    test_eval_labels: np.ndarray
    scaler_payload: dict[str, Any]


@dataclass(frozen=True)
class SMDScoreBundle:
    """Model scores before thresholding."""

    model_name: SMDModelName
    train_scores: np.ndarray
    validation_scores: np.ndarray
    test_scores: np.ndarray
    per_metric_scores: np.ndarray | None
    training_seconds: float
    inference_seconds: float
    parameter_count: int | None
    peak_rss_mb: float | None
    train_rows: int
    validation_rows: int
    test_rows: int


@dataclass(frozen=True)
class SMDThresholdedOutcome:
    """Scores, threshold, and predictions for one model-policy pair."""

    bundle: SMDScoreBundle
    strategy: ThresholdStrategy
    threshold: float
    predictions: np.ndarray
    metrics: BinaryAnomalyMetrics
    delay: DetectionDelaySummary


@dataclass(frozen=True)
class SMDMultivariateExperimentResult:
    """Location and summary for a completed Phase 6 run."""

    run_dir: Path
    dataset_audit: dict[str, Any]
    metrics: dict[str, Any]


def load_smd_multivariate_config(path: Path) -> SMDMultivariateExperimentConfig:
    """Load a Phase 6 SMD experiment config from YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a YAML mapping")
    default = SMDMultivariateExperimentConfig()
    dense_payload = asdict(default.dense_autoencoder_config)
    dense_payload.update(_mapping(payload.get("dense_autoencoder_config")))
    lstm_payload = asdict(default.lstm_autoencoder_config)
    lstm_payload.update(_mapping(payload.get("lstm_autoencoder_config")))
    isolation_payload = asdict(default.isolation_forest_config)
    isolation_payload.update(_mapping(payload.get("isolation_forest_config")))
    sensitivity_payload = _mapping(payload.get("window_sensitivity"))
    sensitivity = SMDWindowSensitivityConfig(
        enabled=bool(sensitivity_payload.get("enabled", default.window_sensitivity.enabled)),
        max_machines=int(
            sensitivity_payload.get("max_machines", default.window_sensitivity.max_machines)
        ),
        window_sizes=_tuple_ints(
            sensitivity_payload.get("window_sizes"),
            default.window_sensitivity.window_sizes,
        ),
        epochs=int(sensitivity_payload.get("epochs", default.window_sensitivity.epochs)),
    )
    return SMDMultivariateExperimentConfig(
        train_fraction=float(payload.get("train_fraction", default.train_fraction)),
        window_size=int(payload.get("window_size", default.window_size)),
        stride=int(payload.get("stride", default.stride)),
        metric_count=int(payload.get("metric_count", default.metric_count)),
        scaler_epsilon=float(payload.get("scaler_epsilon", default.scaler_epsilon)),
        threshold_percentile=float(
            payload.get("threshold_percentile", default.threshold_percentile)
        ),
        threshold_strategies=_threshold_strategies(
            payload.get("threshold_strategies"),
            default.threshold_strategies,
        ),
        max_machines=(
            int(payload["max_machines"]) if payload.get("max_machines") is not None else None
        ),
        train_row_limit=(
            int(payload["train_row_limit"])
            if payload.get("train_row_limit") is not None
            else default.train_row_limit
        ),
        train_window_limit=(
            int(payload["train_window_limit"])
            if payload.get("train_window_limit") is not None
            else default.train_window_limit
        ),
        top_k_metrics=int(payload.get("top_k_metrics", default.top_k_metrics)),
        max_top_metric_rows_per_outcome=int(
            payload.get(
                "max_top_metric_rows_per_outcome",
                default.max_top_metric_rows_per_outcome,
            )
        ),
        max_visualized_machines=int(
            payload.get("max_visualized_machines", default.max_visualized_machines)
        ),
        random_seed=int(payload.get("random_seed", default.random_seed)),
        isolation_forest_config=IsolationForestConfig(**isolation_payload),
        dense_autoencoder_config=DenseAutoencoderConfig.from_dict(dense_payload),
        lstm_autoencoder_config=LSTMAutoencoderConfig.from_dict(lstm_payload),
        window_sensitivity=sensitivity,
    )


def load_smd_machines(
    root: Path,
    *,
    metric_count: int = 38,
    max_machines: int | None = None,
) -> list[SMDMachineData]:
    """Load raw SMD matrices from disk while preserving machine identity."""

    raw_root = root / "data" / "raw" / "smd"
    train_dir = raw_root / "train"
    test_dir = raw_root / "test"
    label_dir = raw_root / "test_label"
    interpretation_dir = raw_root / "interpretation_label"
    if not train_dir.exists():
        raise FileNotFoundError(f"SMD train directory not found: {train_dir}")

    train_files = sorted(train_dir.glob("*.txt"))
    if max_machines is not None:
        train_files = train_files[:max_machines]
    if not train_files:
        raise FileNotFoundError(f"no SMD train files found under: {train_dir}")

    metric_names = tuple(f"metric_{index:02d}" for index in range(metric_count))
    machines: list[SMDMachineData] = []
    for train_file in train_files:
        machine_id = train_file.stem
        test_file = test_dir / train_file.name
        label_file = label_dir / train_file.name
        interpretation_file = interpretation_dir / train_file.name
        if not test_file.exists():
            raise FileNotFoundError(f"missing SMD test file for {machine_id}: {test_file}")
        if not label_file.exists():
            raise FileNotFoundError(f"missing SMD label file for {machine_id}: {label_file}")

        train = _load_smd_matrix(train_file, metric_count=metric_count)
        test = _load_smd_matrix(test_file, metric_count=metric_count)
        labels = np.asarray([int(value) for value in read_non_empty_lines(label_file)], dtype=int)
        if labels.shape[0] != test.shape[0]:
            raise ValueError(
                f"SMD test/label length mismatch for {machine_id}: "
                f"test={test.shape[0]}, labels={labels.shape[0]}"
            )
        if set(np.unique(labels).tolist()).difference({0, 1}):
            raise ValueError(f"SMD labels must be binary for {machine_id}")
        machines.append(
            SMDMachineData(
                machine_id=machine_id,
                train=train,
                test=test,
                test_labels=labels,
                interpretation_intervals=tuple(_load_interpretation(interpretation_file)),
                metric_names=metric_names,
            )
        )
    return machines


def audit_smd_machines(
    machines: list[SMDMachineData],
    *,
    root: Path | None = None,
    metric_count: int = 38,
    near_constant_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Audit the exact SMD matrices used by the experiment."""

    if not machines:
        raise ValueError("cannot audit an empty SMD machine list")

    machine_rows: list[dict[str, Any]] = []
    for machine in machines:
        train_constant = _constant_feature_indices(machine.train)
        test_constant = _constant_feature_indices(machine.test)
        train_near_constant = _constant_feature_indices(
            machine.train,
            epsilon=near_constant_epsilon,
        )
        test_near_constant = _constant_feature_indices(
            machine.test,
            epsilon=near_constant_epsilon,
        )
        segments = positive_segments(machine.test_labels)
        machine_rows.append(
            {
                "machine_id": machine.machine_id,
                "train_rows": int(machine.train.shape[0]),
                "test_rows": int(machine.test.shape[0]),
                "feature_count": int(machine.train.shape[1]),
                "label_rows": int(machine.test_labels.shape[0]),
                "positive_test_labels": int(machine.test_labels.sum()),
                "positive_label_rate": float(machine.test_labels.mean()),
                "anomaly_segments": len(segments),
                "max_anomaly_segment_length": max(
                    (end - start for start, end in segments),
                    default=0,
                ),
                "interpretation_intervals": len(machine.interpretation_intervals),
                "affected_metric_mentions": sum(
                    len(interval.affected_metric_indices)
                    for interval in machine.interpretation_intervals
                ),
                "train_missing_values": int(np.count_nonzero(~np.isfinite(machine.train))),
                "test_missing_values": int(np.count_nonzero(~np.isfinite(machine.test))),
                "train_constant_feature_count": len(train_constant),
                "test_constant_feature_count": len(test_constant),
                "train_constant_features": train_constant,
                "test_constant_features": test_constant,
                "train_near_constant_epsilon": near_constant_epsilon,
                "test_near_constant_epsilon": near_constant_epsilon,
                "train_near_constant_feature_count": len(train_near_constant),
                "test_near_constant_feature_count": len(test_near_constant),
                "train_near_constant_features": train_near_constant,
                "test_near_constant_features": test_near_constant,
                "train_duplicate_rows": _duplicate_row_count(machine.train),
                "test_duplicate_rows": _duplicate_row_count(machine.test),
                "wide_float32_memory_mib": _matrix_mib(machine.train, machine.test, bytes_per=4),
                "wide_float64_memory_mib": _matrix_mib(machine.train, machine.test, bytes_per=8),
            }
        )

    frame = pd.DataFrame(machine_rows)
    labels = np.concatenate([machine.test_labels for machine in machines])
    profile: dict[str, Any] | None = None
    profile_error: str | None = None
    if root is not None:
        try:
            profile = profile_smd_dataset(root, SMDProfileConfig(metric_count=metric_count))
        except Exception as exc:  # pragma: no cover - audit must not hide matrix-level facts
            profile_error = str(exc)

    return {
        "dataset_id": "smd",
        "machine_count": len(machines),
        "metric_count": metric_count,
        "temporal_axis": "row_index",
        "timestamp_column": False,
        "train_rows": int(frame["train_rows"].sum()),
        "test_rows": int(frame["test_rows"].sum()),
        "label_rows": int(frame["label_rows"].sum()),
        "positive_test_labels": int(labels.sum()),
        "positive_label_rate": float(labels.mean()),
        "machines_with_anomalies": int((frame["positive_test_labels"] > 0).sum()),
        "anomaly_segments": int(frame["anomaly_segments"].sum()),
        "interpretation_intervals": int(frame["interpretation_intervals"].sum()),
        "missing_values": int(
            frame["train_missing_values"].sum() + frame["test_missing_values"].sum()
        ),
        "train_rows_min": int(frame["train_rows"].min()),
        "train_rows_max": int(frame["train_rows"].max()),
        "test_rows_min": int(frame["test_rows"].min()),
        "test_rows_max": int(frame["test_rows"].max()),
        "machines_with_train_constant_features": int(
            (frame["train_constant_feature_count"] > 0).sum()
        ),
        "machines_with_test_constant_features": int(
            (frame["test_constant_feature_count"] > 0).sum()
        ),
        "near_constant_epsilon": near_constant_epsilon,
        "machines_with_train_near_constant_features": int(
            (frame["train_near_constant_feature_count"] > 0).sum()
        ),
        "machines_with_test_near_constant_features": int(
            (frame["test_near_constant_feature_count"] > 0).sum()
        ),
        "train_duplicate_rows": int(frame["train_duplicate_rows"].sum()),
        "test_duplicate_rows": int(frame["test_duplicate_rows"].sum()),
        "wide_float32_memory_mib": float(frame["wide_float32_memory_mib"].sum()),
        "wide_float64_memory_mib": float(frame["wide_float64_memory_mib"].sum()),
        "processed_profile": profile,
        "processed_profile_error": profile_error,
        "machine_audit": machine_rows,
    }


def prepare_smd_machine(
    machine: SMDMachineData,
    config: SMDMultivariateExperimentConfig,
) -> SMDPreparedMachine:
    """Create split-local scaled matrices and sequence windows for one machine."""

    train_rows = int(machine.train.shape[0])
    train_end = int(np.floor(train_rows * config.train_fraction))
    train_end = min(max(train_end, config.window_size), train_rows - config.window_size)
    if train_end < config.window_size or train_rows - train_end < config.window_size:
        raise ValueError(
            f"{machine.machine_id} does not have enough rows for "
            f"window_size={config.window_size} and validation split"
        )

    train_fit = machine.train[:train_end]
    validation = machine.train[train_end:]
    scaler = TimeSeriesStandardScaler(epsilon=config.scaler_epsilon).fit(train_fit)
    train_fit_scaled = scaler.transform(train_fit).astype(np.float32)
    validation_scaled = scaler.transform(validation).astype(np.float32)
    test_scaled = scaler.transform(machine.test).astype(np.float32)

    window_config = SequenceWindowConfig(
        sequence_length=config.window_size,
        stride=config.stride,
    )
    train_windows = make_sequence_windows(train_fit_scaled, config=window_config)
    validation_windows = make_sequence_windows(validation_scaled, config=window_config)
    test_windows = make_sequence_windows(
        test_scaled,
        labels=machine.test_labels,
        config=window_config,
    )
    if train_windows.is_empty or validation_windows.is_empty or test_windows.is_empty:
        raise ValueError(f"{machine.machine_id} produced empty SMD windows")

    endpoint_indices = test_windows.end_indices.astype(int)
    return SMDPreparedMachine(
        machine=machine,
        train_fit=train_fit,
        validation=validation,
        test=machine.test,
        train_fit_scaled=train_fit_scaled,
        validation_scaled=validation_scaled,
        test_scaled=test_scaled,
        train_windows=train_windows,
        validation_windows=validation_windows,
        test_windows=test_windows,
        test_eval_indices=endpoint_indices,
        test_eval_labels=machine.test_labels[endpoint_indices].astype(int),
        scaler_payload=scaler.to_dict(),
    )


def threshold_from_scores(
    *,
    train_scores: np.ndarray,
    validation_scores: np.ndarray,
    strategy: ThresholdStrategy,
    percentile: float,
) -> float:
    """Select a threshold from train or validation scores only."""

    if strategy == "train_percentile":
        return quantile_threshold(train_scores, percentile / 100)
    if strategy == "validation_percentile":
        return quantile_threshold(validation_scores, percentile / 100)
    raise ValueError(f"unsupported threshold strategy: {strategy}")


def score_per_feature_zscore(prepared: SMDPreparedMachine) -> SMDScoreBundle:
    """Score each endpoint by the largest absolute train-scaled feature value."""

    started = time.perf_counter()
    train_feature_scores = np.abs(prepared.train_fit_scaled[prepared.train_windows.end_indices])
    validation_feature_scores = np.abs(
        prepared.validation_scaled[prepared.validation_windows.end_indices]
    )
    test_feature_scores = np.abs(prepared.test_scaled[prepared.test_eval_indices])
    inference_seconds = float(time.perf_counter() - started)
    return SMDScoreBundle(
        model_name="per_feature_zscore",
        train_scores=train_feature_scores.max(axis=1).astype(float),
        validation_scores=validation_feature_scores.max(axis=1).astype(float),
        test_scores=test_feature_scores.max(axis=1).astype(float),
        per_metric_scores=test_feature_scores.astype(np.float32),
        training_seconds=0.0,
        inference_seconds=inference_seconds,
        parameter_count=0,
        peak_rss_mb=process_rss_mb(),
        train_rows=int(prepared.train_fit_scaled.shape[0]),
        validation_rows=int(prepared.validation_scaled.shape[0]),
        test_rows=int(prepared.test_scaled.shape[0]),
    )


def score_isolation_forest(
    prepared: SMDPreparedMachine,
    *,
    config: SMDMultivariateExperimentConfig,
    model_dir: Path | None = None,
) -> SMDScoreBundle:
    """Fit a point-wise multivariate Isolation Forest and score window endpoints."""

    fit_matrix = _limit_rows_evenly(prepared.train_fit_scaled, config.train_row_limit)
    detector = IsolationForestDetector(config.isolation_forest_config)
    started = time.perf_counter()
    detector.fit(fit_matrix)
    training_seconds = float(time.perf_counter() - started)

    started = time.perf_counter()
    train_scores = detector.score_samples(
        prepared.train_fit_scaled[prepared.train_windows.end_indices]
    )
    validation_scores = detector.score_samples(
        prepared.validation_scaled[prepared.validation_windows.end_indices]
    )
    test_scores = detector.score_samples(prepared.test_scaled[prepared.test_eval_indices])
    inference_seconds = float(time.perf_counter() - started)
    if model_dir is not None:
        detector.save(model_dir / f"{prepared.machine.machine_id}.joblib")
    return SMDScoreBundle(
        model_name="isolation_forest",
        train_scores=train_scores.astype(float),
        validation_scores=validation_scores.astype(float),
        test_scores=test_scores.astype(float),
        per_metric_scores=None,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=None,
        peak_rss_mb=process_rss_mb(),
        train_rows=int(fit_matrix.shape[0]),
        validation_rows=int(prepared.validation_scaled.shape[0]),
        test_rows=int(prepared.test_scaled.shape[0]),
    )


def score_dense_autoencoder(
    prepared: SMDPreparedMachine,
    *,
    config: SMDMultivariateExperimentConfig,
    model_dir: Path | None = None,
) -> SMDScoreBundle:
    """Fit and score a multivariate dense autoencoder on SMD windows."""

    train_windows = _limit_rows_evenly(
        prepared.train_windows.windows,
        config.train_window_limit,
    )
    detector = DenseAutoencoderDetector(config.dense_autoencoder_config)
    checkpoint_path = (
        model_dir / f"{prepared.machine.machine_id}.checkpoint.pt" if model_dir else None
    )
    summary = detector.fit(
        train_windows,
        validation_windows=prepared.validation_windows.windows,
        checkpoint_path=checkpoint_path,
    )

    started = time.perf_counter()
    train_scores = detector.reconstruction_error(prepared.train_windows.windows)
    validation_scores = detector.reconstruction_error(prepared.validation_windows.windows)
    per_metric_scores = detector.reconstruction_error_by_feature(prepared.test_windows.windows)
    test_scores = per_metric_scores.mean(axis=1)
    inference_seconds = float(time.perf_counter() - started)
    if model_dir is not None:
        detector.save(model_dir / f"{prepared.machine.machine_id}.pt")
    return SMDScoreBundle(
        model_name="dense_autoencoder",
        train_scores=train_scores.astype(float),
        validation_scores=validation_scores.astype(float),
        test_scores=test_scores.astype(float),
        per_metric_scores=per_metric_scores.astype(np.float32),
        training_seconds=summary.training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=summary.parameter_count,
        peak_rss_mb=summary.peak_rss_mb,
        train_rows=int(train_windows.shape[0]),
        validation_rows=int(prepared.validation_windows.windows.shape[0]),
        test_rows=int(prepared.test_windows.windows.shape[0]),
    )


def score_lstm_autoencoder(
    prepared: SMDPreparedMachine,
    *,
    config: SMDMultivariateExperimentConfig,
    model_dir: Path | None = None,
) -> SMDScoreBundle:
    """Fit and score a multivariate LSTM autoencoder on SMD windows."""

    train_windows = _limit_rows_evenly(
        prepared.train_windows.windows,
        config.train_window_limit,
    )
    detector = LSTMAutoencoderDetector(config.lstm_autoencoder_config)
    checkpoint_path = (
        model_dir / f"{prepared.machine.machine_id}.checkpoint.pt" if model_dir else None
    )
    summary = detector.fit(
        train_windows,
        validation_windows=prepared.validation_windows.windows,
        checkpoint_path=checkpoint_path,
    )

    started = time.perf_counter()
    train_scores = detector.reconstruction_error(prepared.train_windows.windows)
    validation_scores = detector.reconstruction_error(prepared.validation_windows.windows)
    per_metric_scores = detector.reconstruction_error_by_feature(prepared.test_windows.windows)
    test_scores = per_metric_scores.mean(axis=1)
    inference_seconds = float(time.perf_counter() - started)
    if model_dir is not None:
        detector.save(model_dir / f"{prepared.machine.machine_id}.pt")
    return SMDScoreBundle(
        model_name="lstm_autoencoder",
        train_scores=train_scores.astype(float),
        validation_scores=validation_scores.astype(float),
        test_scores=test_scores.astype(float),
        per_metric_scores=per_metric_scores.astype(np.float32),
        training_seconds=summary.training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=summary.parameter_count,
        peak_rss_mb=summary.peak_rss_mb,
        train_rows=int(train_windows.shape[0]),
        validation_rows=int(prepared.validation_windows.windows.shape[0]),
        test_rows=int(prepared.test_windows.windows.shape[0]),
    )


def evaluate_score_bundle(
    prepared: SMDPreparedMachine,
    bundle: SMDScoreBundle,
    *,
    strategy: ThresholdStrategy,
    threshold_percentile: float,
) -> SMDThresholdedOutcome:
    """Threshold and evaluate one SMD score bundle."""

    threshold = threshold_from_scores(
        train_scores=bundle.train_scores,
        validation_scores=bundle.validation_scores,
        strategy=strategy,
        percentile=threshold_percentile,
    )
    predictions = (bundle.test_scores >= threshold).astype(int)
    metrics = compute_binary_metrics(prepared.test_eval_labels, predictions, bundle.test_scores)
    delay = _compute_delay(prepared.test_eval_labels, predictions)
    return SMDThresholdedOutcome(
        bundle=bundle,
        strategy=strategy,
        threshold=threshold,
        predictions=predictions,
        metrics=metrics,
        delay=delay,
    )


def run_smd_multivariate_experiment(
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    config: SMDMultivariateExperimentConfig | None = None,
) -> SMDMultivariateExperimentResult:
    """Run the Phase 6 SMD multivariate anomaly-detection experiment."""

    cfg = config or SMDMultivariateExperimentConfig()
    project = project_root() if root is None else root
    set_reproducible_seed(cfg.random_seed)
    resolved_run_id = run_id or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    base_output = output_root or project / "experiments" / "anomaly" / "smd"
    run_dir = base_output / resolved_run_id
    figures_dir = run_dir / "figures"
    models_dir = run_dir / "models"
    scores_dir = run_dir / "per_metric_scores"
    for directory in (run_dir, figures_dir, models_dir, scores_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    machines = load_smd_machines(
        project,
        metric_count=cfg.metric_count,
        max_machines=cfg.max_machines,
    )
    audit = audit_smd_machines(
        machines,
        root=project,
        metric_count=cfg.metric_count,
        near_constant_epsilon=cfg.scaler_epsilon,
    )
    _save_json(run_dir / "config.json", asdict(cfg))
    _save_json(run_dir / "dataset_audit.json", audit)
    pd.DataFrame(audit["machine_audit"]).to_csv(
        run_dir / "dataset_audit_by_machine.csv", index=False
    )

    per_machine_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    top_metric_rows: list[dict[str, Any]] = []
    plotted_machines = 0

    for machine_index, machine in enumerate(machines):
        machine_started = time.perf_counter()
        prepared = prepare_smd_machine(machine, cfg)
        machine_model_dir = models_dir
        bundles = [
            score_per_feature_zscore(prepared),
            score_isolation_forest(
                prepared,
                config=cfg,
                model_dir=machine_model_dir / "isolation_forest",
            ),
            score_dense_autoencoder(
                prepared,
                config=cfg,
                model_dir=machine_model_dir / "dense_autoencoder",
            ),
            score_lstm_autoencoder(
                prepared,
                config=cfg,
                model_dir=machine_model_dir / "lstm_autoencoder",
            ),
        ]

        local_outcomes: dict[tuple[SMDModelName, ThresholdStrategy], SMDThresholdedOutcome] = {}
        for bundle in bundles:
            runtime_rows.append(
                {
                    "machine_id": machine.machine_id,
                    "model": bundle.model_name,
                    "training_seconds": bundle.training_seconds,
                    "inference_seconds": bundle.inference_seconds,
                    "parameter_count": bundle.parameter_count,
                    "peak_rss_mb": bundle.peak_rss_mb,
                    "train_rows_used": bundle.train_rows,
                    "validation_rows_scored": bundle.validation_rows,
                    "test_rows_scored": bundle.test_rows,
                    "window_size": cfg.window_size,
                    "stride": cfg.stride,
                    "machine_runtime_seconds": float(time.perf_counter() - machine_started),
                }
            )
            if bundle.per_metric_scores is not None:
                _save_per_metric_scores(scores_dir, machine, prepared, bundle)

            for strategy in cfg.threshold_strategies:
                outcome = evaluate_score_bundle(
                    prepared,
                    bundle,
                    strategy=strategy,
                    threshold_percentile=cfg.threshold_percentile,
                )
                local_outcomes[(bundle.model_name, strategy)] = outcome
                per_machine_rows.append(_per_machine_metrics_row(machine, prepared, outcome, cfg))
                prediction_frames.append(_prediction_frame(machine, prepared, outcome))
                top_metric_rows.extend(
                    _top_metric_records(
                        machine=machine,
                        prepared=prepared,
                        outcome=outcome,
                        top_k=cfg.top_k_metrics,
                        max_rows=cfg.max_top_metric_rows_per_outcome,
                    )
                )

        if plotted_machines < cfg.max_visualized_machines:
            best_outcome = max(
                local_outcomes.values(),
                key=lambda outcome: (
                    outcome.metrics.f1,
                    outcome.metrics.recall,
                    -outcome.metrics.false_positive_rate,
                ),
            )
            _plot_machine_overview(
                prepared,
                best_outcome,
                figures_dir=figures_dir,
                machine_index=machine_index,
            )
            plotted_machines += 1

    per_machine_df = pd.DataFrame(per_machine_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    predictions_df = _finalize_predictions(prediction_frames)
    top_metrics_df = pd.DataFrame(top_metric_rows)
    if not top_metrics_df.empty:
        top_metrics_df.to_csv(run_dir / "per_metric_top_signals.csv", index=False)
        _metric_frequency(top_metrics_df).to_csv(
            run_dir / "per_metric_signal_frequency.csv",
            index=False,
        )
        _plot_top_metric_frequency(top_metrics_df, figures_dir)

    global_df = _global_metrics(predictions_df)
    per_machine_summary = _per_machine_summary(per_machine_df, machine_count=len(machines))
    benefit_df, benefit_summary = _multivariate_benefit(per_machine_df)
    best_per_machine = _best_model_per_machine(per_machine_df)
    error_analysis = _error_analysis(per_machine_df, global_df, benefit_summary)
    sensitivity_df = _run_window_sensitivity(
        machines,
        config=cfg,
        run_dir=run_dir,
    )

    per_machine_df.to_csv(run_dir / "per_machine_metrics.csv", index=False)
    runtime_df.to_csv(run_dir / "runtime.csv", index=False)
    global_df.to_csv(run_dir / "global_metrics.csv", index=False)
    pd.DataFrame(per_machine_summary).to_csv(run_dir / "per_machine_summary.csv", index=False)
    benefit_df.to_csv(run_dir / "multivariate_benefit_by_machine.csv", index=False)
    best_per_machine.to_csv(run_dir / "best_model_per_machine.csv", index=False)
    if not sensitivity_df.empty:
        sensitivity_df.to_csv(run_dir / "window_sensitivity.csv", index=False)
    predictions_path = _write_dataframe(predictions_df, run_dir / "predictions")
    predictions_df.head(5000).to_csv(run_dir / "prediction_sample.csv", index=False)

    metrics = {
        "global": _records_by_key(global_df, keys=("model", "threshold_strategy")),
        "per_machine_summary": per_machine_summary,
        "best_model": _best_global_model(global_df),
        "best_model_per_machine_path": "best_model_per_machine.csv",
        "multivariate_benefit": benefit_summary,
        "computational_efficiency": _computational_efficiency(runtime_df),
        "error_analysis": error_analysis,
        "window_sensitivity_rows": int(sensitivity_df.shape[0]),
        "prediction_artifact": str(predictions_path.relative_to(run_dir)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _save_json(run_dir / "metrics.json", metrics)
    _save_json(run_dir / "error_analysis.json", error_analysis)
    _plot_model_comparison(global_df, figures_dir)
    _plot_per_machine_f1_distribution(per_machine_df, figures_dir)
    _write_run_report(run_dir, audit=audit, metrics=metrics, config=cfg)
    return SMDMultivariateExperimentResult(run_dir=run_dir, dataset_audit=audit, metrics=metrics)


def _load_smd_matrix(path: Path, *, metric_count: int) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    if matrix.ndim != 2:
        raise ValueError(f"SMD matrix must be 2D: {path}")
    if matrix.shape[1] != metric_count:
        raise ValueError(
            f"SMD feature mismatch for {path}: expected {metric_count}, found {matrix.shape[1]}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"SMD matrix contains NaN or infinite values: {path}")
    return matrix


def _load_interpretation(path: Path) -> list[SMDInterpretationInterval]:
    if not path.exists():
        return []
    intervals: list[SMDInterpretationInterval] = []
    for line in read_non_empty_lines(path):
        interval, metrics = parse_interpretation_line(line)
        if interval is None:
            continue
        intervals.append(
            SMDInterpretationInterval(
                start_index=interval[0],
                end_index=interval[1],
                affected_metric_indices=tuple(metrics),
            )
        )
    return intervals


def _constant_feature_indices(matrix: np.ndarray, *, epsilon: float = 1e-12) -> list[int]:
    deviations = np.std(matrix, axis=0)
    return [int(index) for index in np.flatnonzero(deviations <= epsilon)]


def _duplicate_row_count(matrix: np.ndarray) -> int:
    if matrix.shape[0] == 0:
        return 0
    return int(pd.DataFrame(matrix).duplicated().sum())


def _matrix_mib(train: np.ndarray, test: np.ndarray, *, bytes_per: int) -> float:
    return float((train.size + test.size) * bytes_per / (1024 * 1024))


def _limit_rows_evenly(array: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or array.shape[0] <= limit:
        return array
    indices = np.linspace(0, array.shape[0] - 1, limit, dtype=int)
    return array[indices]


def _compute_delay(y_true: np.ndarray, y_pred: np.ndarray) -> DetectionDelaySummary:
    delay_steps: list[int] = []
    anomaly_windows = 0
    for start, end in positive_segments(y_true):
        anomaly_windows += 1
        local_hits = np.flatnonzero(y_pred[start:end] == 1)
        if local_hits.size == 0:
            continue
        delay_steps.append(int(local_hits[0]))
    return DetectionDelaySummary(
        anomaly_windows=anomaly_windows,
        detected_windows=len(delay_steps),
        missed_windows=anomaly_windows - len(delay_steps),
        mean_delay_steps=float(np.mean(delay_steps)) if delay_steps else None,
        median_delay_steps=float(np.median(delay_steps)) if delay_steps else None,
        mean_delay_seconds=None,
        median_delay_seconds=None,
    )


def _per_machine_metrics_row(
    machine: SMDMachineData,
    prepared: SMDPreparedMachine,
    outcome: SMDThresholdedOutcome,
    config: SMDMultivariateExperimentConfig,
) -> dict[str, Any]:
    metrics = outcome.metrics.to_dict()
    delay = outcome.delay.to_dict()
    return {
        "machine_id": machine.machine_id,
        "model": outcome.bundle.model_name,
        "threshold_strategy": outcome.strategy,
        "threshold_percentile": config.threshold_percentile,
        "threshold": outcome.threshold,
        "window_size": config.window_size,
        "stride": config.stride,
        "train_rows": int(prepared.train_fit.shape[0]),
        "validation_rows": int(prepared.validation.shape[0]),
        "test_eval_rows": int(prepared.test_eval_labels.shape[0]),
        **metrics,
        **{f"detection_{key}": value for key, value in delay.items()},
    }


def _prediction_frame(
    machine: SMDMachineData,
    prepared: SMDPreparedMachine,
    outcome: SMDThresholdedOutcome,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": machine.machine_id,
            "model": outcome.bundle.model_name,
            "threshold_strategy": outcome.strategy,
            "sequence_index": prepared.test_eval_indices.astype(np.int32),
            "y_true": prepared.test_eval_labels.astype(np.int8),
            "score": outcome.bundle.test_scores.astype(np.float32),
            "threshold": np.full(
                prepared.test_eval_indices.shape[0],
                outcome.threshold,
                dtype=np.float32,
            ),
            "y_pred": outcome.predictions.astype(np.int8),
        }
    )


def _finalize_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("no prediction frames were generated")
    frame = pd.concat(frames, ignore_index=True)
    for column in ("machine_id", "model", "threshold_strategy"):
        frame[column] = frame[column].astype("category")
    return frame


def _save_per_metric_scores(
    scores_dir: Path,
    machine: SMDMachineData,
    prepared: SMDPreparedMachine,
    bundle: SMDScoreBundle,
) -> None:
    if bundle.per_metric_scores is None:
        return
    output_dir = scores_dir / bundle.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{machine.machine_id}.npz",
        machine_id=np.asarray(machine.machine_id),
        metric_names=np.asarray(machine.metric_names),
        sequence_index=prepared.test_eval_indices.astype(np.int32),
        y_true=prepared.test_eval_labels.astype(np.int8),
        global_score=bundle.test_scores.astype(np.float32),
        per_metric_score=bundle.per_metric_scores.astype(np.float32),
    )


def _top_metric_records(
    *,
    machine: SMDMachineData,
    prepared: SMDPreparedMachine,
    outcome: SMDThresholdedOutcome,
    top_k: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    per_metric = outcome.bundle.per_metric_scores
    if per_metric is None or per_metric.size == 0:
        return []
    candidate_mask = outcome.predictions.astype(bool)
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        candidate_indices = np.argsort(-outcome.bundle.test_scores)[:max_rows]
    else:
        ordered = candidate_indices[np.argsort(-outcome.bundle.test_scores[candidate_indices])]
        candidate_indices = ordered[:max_rows]
    records: list[dict[str, Any]] = []
    for row_index in candidate_indices:
        metric_order = np.argsort(-per_metric[row_index])[:top_k]
        for rank, metric_index in enumerate(metric_order, start=1):
            records.append(
                {
                    "machine_id": machine.machine_id,
                    "model": outcome.bundle.model_name,
                    "threshold_strategy": outcome.strategy,
                    "sequence_index": int(prepared.test_eval_indices[row_index]),
                    "y_true": int(prepared.test_eval_labels[row_index]),
                    "y_pred": int(outcome.predictions[row_index]),
                    "global_score": float(outcome.bundle.test_scores[row_index]),
                    "threshold": float(outcome.threshold),
                    "rank": rank,
                    "metric_index": int(metric_index),
                    "metric_name": machine.metric_names[int(metric_index)],
                    "per_metric_score": float(per_metric[row_index, metric_index]),
                    "evidence_kind": (
                        "absolute_train_scaled_value"
                        if outcome.bundle.model_name == "per_feature_zscore"
                        else "reconstruction_error"
                    ),
                }
            )
    return records


def _metric_frequency(top_metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        top_metrics_df.groupby(
            ["model", "threshold_strategy", "machine_id", "metric_name"],
            observed=True,
        )
        .agg(
            mentions=("metric_name", "size"),
            mean_per_metric_score=("per_metric_score", "mean"),
            mean_global_score=("global_score", "mean"),
        )
        .reset_index()
        .sort_values(["mentions", "mean_per_metric_score"], ascending=[False, False])
    )
    return cast(pd.DataFrame, grouped)


def _global_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, strategy), group in predictions.groupby(
        ["model", "threshold_strategy"],
        observed=True,
    ):
        ordered = group.sort_values(["machine_id", "sequence_index"])
        y_true = ordered["y_true"].to_numpy(dtype=int)
        y_pred = ordered["y_pred"].to_numpy(dtype=int)
        scores = ordered["score"].to_numpy(dtype=float)
        metrics = compute_binary_metrics(y_true, y_pred, scores).to_dict()
        delay = _grouped_sequence_delay(ordered).to_dict()
        rows.append(
            {
                "model": str(model),
                "threshold_strategy": str(strategy),
                **metrics,
                **{f"detection_{key}": value for key, value in delay.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["f1", "recall", "false_positive_rate"],
        ascending=[False, False, True],
    )


def _grouped_sequence_delay(frame: pd.DataFrame) -> DetectionDelaySummary:
    delay_steps: list[int] = []
    anomaly_windows = 0
    for _machine_id, group in frame.groupby("machine_id", observed=True):
        ordered = group.sort_values("sequence_index")
        truth = ordered["y_true"].to_numpy(dtype=int)
        pred = ordered["y_pred"].to_numpy(dtype=int)
        for start, end in positive_segments(truth):
            anomaly_windows += 1
            local_hits = np.flatnonzero(pred[start:end] == 1)
            if local_hits.size:
                delay_steps.append(int(local_hits[0]))
    return DetectionDelaySummary(
        anomaly_windows=anomaly_windows,
        detected_windows=len(delay_steps),
        missed_windows=anomaly_windows - len(delay_steps),
        mean_delay_steps=float(np.mean(delay_steps)) if delay_steps else None,
        median_delay_steps=float(np.median(delay_steps)) if delay_steps else None,
        mean_delay_seconds=None,
        median_delay_seconds=None,
    )


def _per_machine_summary(
    per_machine_df: pd.DataFrame,
    *,
    machine_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, strategy), group in per_machine_df.groupby(
        ["model", "threshold_strategy"],
        observed=True,
    ):
        rows.append(
            {
                "model": str(model),
                "threshold_strategy": str(strategy),
                "machine_count": int(group["machine_id"].nunique()),
                "machine_coverage": float(group["machine_id"].nunique() / machine_count),
                "f1_mean": float(group["f1"].mean()),
                "f1_median": float(group["f1"].median()),
                "f1_std": float(group["f1"].std(ddof=0)),
                "machines_with_f1_zero": int((group["f1"] == 0).sum()),
                "precision_mean": float(group["precision"].mean()),
                "recall_mean": float(group["recall"].mean()),
                "false_positive_rate_mean": float(group["false_positive_rate"].mean()),
            }
        )
    return sorted(rows, key=lambda row: (row["f1_mean"], row["recall_mean"]), reverse=True)


def _multivariate_benefit(per_machine_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline = (
        per_machine_df[per_machine_df["model"].eq("per_feature_zscore")]
        .sort_values(["machine_id", "f1", "recall"], ascending=[True, False, False])
        .groupby("machine_id", observed=True)
        .head(1)
        .loc[:, ["machine_id", "model", "threshold_strategy", "f1", "recall"]]
        .rename(
            columns={
                "model": "baseline_model",
                "threshold_strategy": "baseline_threshold_strategy",
                "f1": "baseline_best_f1",
                "recall": "baseline_best_recall",
            }
        )
    )
    multivariate = (
        per_machine_df[~per_machine_df["model"].eq("per_feature_zscore")]
        .sort_values(["machine_id", "f1", "recall"], ascending=[True, False, False])
        .groupby("machine_id", observed=True)
        .head(1)
        .loc[:, ["machine_id", "model", "threshold_strategy", "f1", "recall"]]
        .rename(
            columns={
                "model": "best_multivariate_model",
                "threshold_strategy": "multivariate_threshold_strategy",
                "f1": "multivariate_best_f1",
                "recall": "multivariate_best_recall",
            }
        )
    )
    comparison = baseline.merge(multivariate, on="machine_id", how="outer")
    comparison["f1_delta_multivariate_minus_zscore"] = (
        comparison["multivariate_best_f1"] - comparison["baseline_best_f1"]
    )
    summary = {
        "mean_f1_delta_multivariate_minus_zscore": float(
            comparison["f1_delta_multivariate_minus_zscore"].mean()
        ),
        "median_f1_delta_multivariate_minus_zscore": float(
            comparison["f1_delta_multivariate_minus_zscore"].median()
        ),
        "machines_multivariate_helped": int(
            (comparison["f1_delta_multivariate_minus_zscore"] > 0).sum()
        ),
        "machines_multivariate_tied": int(
            (comparison["f1_delta_multivariate_minus_zscore"] == 0).sum()
        ),
        "machines_multivariate_hurt": int(
            (comparison["f1_delta_multivariate_minus_zscore"] < 0).sum()
        ),
        "note": (
            "This is an empirical association inside SMD only; it does not establish "
            "causation or direct comparability with NAB."
        ),
    }
    return comparison, summary


def _best_model_per_machine(per_machine_df: pd.DataFrame) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        per_machine_df.sort_values(
            ["machine_id", "f1", "recall", "false_positive_rate"],
            ascending=[True, False, False, True],
        )
        .groupby("machine_id", observed=True)
        .head(1)
        .reset_index(drop=True),
    )


def _best_global_model(global_df: pd.DataFrame) -> dict[str, Any]:
    if global_df.empty:
        return {}
    return cast(dict[str, Any], global_df.iloc[0].to_dict())


def _error_analysis(
    per_machine_df: pd.DataFrame,
    global_df: pd.DataFrame,
    benefit_summary: dict[str, Any],
) -> dict[str, Any]:
    best_global = _best_global_model(global_df)
    if not best_global:
        return {}
    best_machine_rows = per_machine_df[
        per_machine_df["model"].eq(best_global["model"])
        & per_machine_df["threshold_strategy"].eq(best_global["threshold_strategy"])
    ].copy()
    easiest = (
        best_machine_rows.sort_values(["f1", "recall"], ascending=[False, False])
        .head(5)
        .loc[:, ["machine_id", "f1", "precision", "recall", "false_positive_rate"]]
        .to_dict(orient="records")
    )
    hardest = (
        best_machine_rows.sort_values(["f1", "recall"], ascending=[True, True])
        .head(5)
        .loc[:, ["machine_id", "f1", "precision", "recall", "false_positive_rate"]]
        .to_dict(orient="records")
    )
    high_false_positive = (
        best_machine_rows.sort_values("false_positive_rate", ascending=False)
        .head(5)
        .loc[:, ["machine_id", "f1", "false_positive_rate", "false_positives"]]
        .to_dict(orient="records")
    )
    high_false_negative = (
        best_machine_rows.sort_values("false_negatives", ascending=False)
        .head(5)
        .loc[:, ["machine_id", "f1", "false_negatives", "positives"]]
        .to_dict(orient="records")
    )
    return {
        "best_global_model": best_global,
        "easiest_machines_for_best_model": easiest,
        "hardest_machines_for_best_model": hardest,
        "highest_false_positive_machines": high_false_positive,
        "highest_false_negative_machines": high_false_negative,
        "multivariate_benefit_summary": benefit_summary,
        "interpretation_note": (
            "Top metric rankings are candidate contributing signals from score attribution, "
            "not causal root-cause claims."
        ),
    }


def _computational_efficiency(runtime_df: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for model, group in runtime_df.groupby("model", observed=True):
        rows[str(model)] = {
            "total_training_seconds": float(group["training_seconds"].sum()),
            "total_inference_seconds": float(group["inference_seconds"].sum()),
            "mean_training_seconds_per_machine": float(group["training_seconds"].mean()),
            "mean_inference_seconds_per_machine": float(group["inference_seconds"].mean()),
            "max_peak_rss_mb": _none_if_nan(group["peak_rss_mb"].max()),
            "median_parameter_count": _none_if_nan(group["parameter_count"].median()),
            "train_rows_used_total": int(group["train_rows_used"].sum()),
        }
    return rows


def _run_window_sensitivity(
    machines: list[SMDMachineData],
    *,
    config: SMDMultivariateExperimentConfig,
    run_dir: Path,
) -> pd.DataFrame:
    if not config.window_sensitivity.enabled:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    selected = machines[: config.window_sensitivity.max_machines]
    for machine in selected:
        for window_size in config.window_sensitivity.window_sizes:
            local_config = replace(
                config,
                window_size=window_size,
                dense_autoencoder_config=replace(
                    config.dense_autoencoder_config,
                    epochs=config.window_sensitivity.epochs,
                    patience=1,
                ),
                lstm_autoencoder_config=replace(
                    config.lstm_autoencoder_config,
                    epochs=config.window_sensitivity.epochs,
                    patience=1,
                ),
                window_sensitivity=replace(config.window_sensitivity, enabled=False),
            )
            prepared = prepare_smd_machine(machine, local_config)
            for bundle in (
                score_dense_autoencoder(
                    prepared,
                    config=local_config,
                    model_dir=run_dir / "window_sensitivity_models" / "dense_autoencoder",
                ),
                score_lstm_autoencoder(
                    prepared,
                    config=local_config,
                    model_dir=run_dir / "window_sensitivity_models" / "lstm_autoencoder",
                ),
            ):
                outcome = evaluate_score_bundle(
                    prepared,
                    bundle,
                    strategy="train_percentile",
                    threshold_percentile=local_config.threshold_percentile,
                )
                rows.append(
                    {
                        "machine_id": machine.machine_id,
                        "model": bundle.model_name,
                        "window_size": window_size,
                        "threshold_strategy": "train_percentile",
                        "precision": outcome.metrics.precision,
                        "recall": outcome.metrics.recall,
                        "f1": outcome.metrics.f1,
                        "pr_auc": outcome.metrics.pr_auc,
                        "false_positive_rate": outcome.metrics.false_positive_rate,
                        "training_seconds": bundle.training_seconds,
                        "inference_seconds": bundle.inference_seconds,
                    }
                )
    return pd.DataFrame(rows)


def _plot_machine_overview(
    prepared: SMDPreparedMachine,
    outcome: SMDThresholdedOutcome,
    *,
    figures_dir: Path,
    machine_index: int,
) -> None:
    machine = prepared.machine
    x = prepared.test_eval_indices
    y_true = prepared.test_eval_labels
    scores = outcome.bundle.test_scores
    per_metric = outcome.bundle.per_metric_scores
    sampled = _sample_indices(len(x), max_points=1800)
    figure, axes = plt.subplots(4, 1, figsize=(12, 9), constrained_layout=True)

    scaled = prepared.test_scaled[prepared.test_eval_indices][sampled].T
    image = axes[0].imshow(
        np.clip(scaled, -5, 5),
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        extent=[int(x[sampled][0]), int(x[sampled][-1]), 0, len(machine.metric_names) - 1],
    )
    axes[0].set_title(f"{machine.machine_id}: multivariate scaled telemetry")
    axes[0].set_ylabel("metric index")
    figure.colorbar(image, ax=axes[0], label="train-scaled value")

    axes[1].plot(x, scores, color="#2454a6", linewidth=0.8, label="global anomaly score")
    axes[1].axhline(
        outcome.threshold, color="#a83232", linestyle="--", linewidth=1.0, label="threshold"
    )
    _shade_anomaly_regions(axes[1], x, y_true)
    axes[1].set_title(f"{outcome.bundle.model_name} score with ground-truth regions")
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper right")

    axes[2].plot(x, y_true, color="#111111", linewidth=0.8, label="ground truth")
    axes[2].plot(
        x, outcome.predictions, color="#d97706", linewidth=0.8, alpha=0.9, label="detection"
    )
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_title("Window-end labels and detections")
    axes[2].legend(loc="upper right")

    if per_metric is not None:
        metric_image = axes[3].imshow(
            per_metric[sampled].T,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            extent=[int(x[sampled][0]), int(x[sampled][-1]), 0, len(machine.metric_names) - 1],
        )
        axes[3].set_title("Per-metric anomaly evidence")
        figure.colorbar(metric_image, ax=axes[3], label="per-metric score")
    else:
        axes[3].text(
            0.5, 0.5, "Per-metric scores are not available for Isolation Forest", ha="center"
        )
        axes[3].set_axis_off()
    axes[3].set_xlabel("test sequence index")
    axes[3].set_ylabel("metric index")
    output = (
        figures_dir
        / f"{machine_index:02d}_{machine.machine_id}_{outcome.bundle.model_name}_overview.png"
    )
    figure.savefig(output)
    plt.close(figure)


def _shade_anomaly_regions(axis: Axes, x: np.ndarray, y_true: np.ndarray) -> None:
    for start, end in positive_segments(y_true):
        axis.axvspan(int(x[start]), int(x[end - 1]), color="#ef4444", alpha=0.12, linewidth=0)


def _plot_top_metric_frequency(top_metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    frequency = _metric_frequency(top_metrics_df)
    top = (
        frequency.groupby("metric_name", observed=True)["mentions"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )
    if top.empty:
        return
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    top.sort_values().plot(kind="barh", ax=axis, color="#33658a")
    axis.set_title("Most frequent candidate contributing metrics")
    axis.set_xlabel("top-k mentions")
    figure.savefig(figures_dir / "top_metric_frequency.png")
    plt.close(figure)


def _plot_model_comparison(global_df: pd.DataFrame, figures_dir: Path) -> None:
    if global_df.empty:
        return
    frame = global_df.copy()
    frame["label"] = frame["model"] + "\n" + frame["threshold_strategy"]
    figure, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    axis.bar(frame["label"], frame["f1"], color="#2f855a")
    axis.set_title("SMD global F1 by model and threshold policy")
    axis.set_ylabel("F1")
    axis.tick_params(axis="x", labelrotation=35)
    figure.savefig(figures_dir / "model_comparison_global_f1.png")
    plt.close(figure)


def _plot_per_machine_f1_distribution(per_machine_df: pd.DataFrame, figures_dir: Path) -> None:
    if per_machine_df.empty:
        return
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    labels: list[str] = []
    values: list[np.ndarray] = []
    for (model, strategy), group in per_machine_df.groupby(
        ["model", "threshold_strategy"],
        observed=True,
    ):
        labels.append(f"{model}\n{strategy}")
        values.append(group["f1"].to_numpy(dtype=float))
    axis.boxplot(values, tick_labels=labels, showmeans=True)
    axis.set_title("Per-machine F1 distribution")
    axis.set_ylabel("F1")
    axis.tick_params(axis="x", labelrotation=35)
    figure.savefig(figures_dir / "per_machine_f1_distribution.png")
    plt.close(figure)


def _sample_indices(length: int, *, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, max_points, dtype=int)


def _write_dataframe(frame: pd.DataFrame, path_without_suffix: Path) -> Path:
    parquet_path = path_without_suffix.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = path_without_suffix.with_suffix(".csv.gz")
        frame.to_csv(csv_path, index=False)
        return csv_path


def _records_by_key(frame: pd.DataFrame, *, keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        key = "|".join(str(record[column]) for column in keys)
        records[key] = cast(dict[str, Any], record)
    return records


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    return value


def _none_if_nan(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping")
    return dict(value)


def _tuple_ints(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of integers")
    return tuple(int(item) for item in value)


def _threshold_strategies(
    value: Any,
    default: tuple[ThresholdStrategy, ...],
) -> tuple[ThresholdStrategy, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError("threshold_strategies must be a list")
    strategies = tuple(cast(ThresholdStrategy, str(item)) for item in value)
    invalid = set(strategies).difference(THRESHOLD_STRATEGIES)
    if invalid:
        raise ValueError(f"unsupported threshold strategies: {sorted(invalid)}")
    return strategies


def _write_run_report(
    run_dir: Path,
    *,
    audit: dict[str, Any],
    metrics: dict[str, Any],
    config: SMDMultivariateExperimentConfig,
) -> None:
    best = metrics.get("best_model", {})
    benefit = metrics.get("multivariate_benefit", {})
    lines = [
        "# SMD Multivariate Anomaly Experiment",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Dataset",
        "",
        f"- Machines evaluated: {audit['machine_count']}",
        f"- Metrics per machine: {audit['metric_count']}",
        f"- Train rows: {audit['train_rows']}",
        f"- Test rows: {audit['test_rows']}",
        f"- Positive test labels: {audit['positive_test_labels']} "
        f"({audit['positive_label_rate']:.4f})",
        f"- Anomaly segments: {audit['anomaly_segments']}",
        f"- Machines with train near-constant features at scaler epsilon: "
        f"{audit['machines_with_train_near_constant_features']}",
        "",
        "## Protocol",
        "",
        f"- Window size: {config.window_size}",
        f"- Stride: {config.stride}",
        f"- Scaler epsilon: {config.scaler_epsilon}",
        f"- Train/validation split inside source train: {config.train_fraction:.2f} / "
        f"{1 - config.train_fraction:.2f}",
        f"- Threshold percentile: {config.threshold_percentile}",
        "- Thresholds are selected from train or validation scores only.",
        "- Test labels are used only for final evaluation.",
        "- Window scores are evaluated against the label at the causal window endpoint.",
        "",
        "## Best Measured Model",
        "",
        f"- Model: {best.get('model')}",
        f"- Threshold strategy: {best.get('threshold_strategy')}",
        f"- Precision: {_format_optional(best.get('precision'))}",
        f"- Recall: {_format_optional(best.get('recall'))}",
        f"- F1: {_format_optional(best.get('f1'))}",
        f"- PR-AUC: {_format_optional(best.get('pr_auc'))}",
        f"- False positive rate: {_format_optional(best.get('false_positive_rate'))}",
        "",
        "## Multivariate Benefit",
        "",
        f"- Mean F1 delta vs per-feature z-score: "
        f"{_format_optional(benefit.get('mean_f1_delta_multivariate_minus_zscore'))}",
        f"- Machines helped: {benefit.get('machines_multivariate_helped')}",
        f"- Machines tied: {benefit.get('machines_multivariate_tied')}",
        f"- Machines hurt: {benefit.get('machines_multivariate_hurt')}",
        "",
        "## Artifacts",
        "",
        "- `global_metrics.csv`",
        "- `per_machine_metrics.csv`",
        "- `best_model_per_machine.csv`",
        "- `multivariate_benefit_by_machine.csv`",
        "- `per_metric_top_signals.csv`",
        "- `per_metric_scores/`",
        "- `figures/`",
        "- `runtime.csv`",
        "",
        "Top metric rows are candidate contributing signals, not causal root causes.",
    ]
    (run_dir / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_optional(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"
