"""Phase 4 deep-learning anomaly detection experiment on NAB."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from aegis_ai.data.dataset_registry import project_root
from aegis_ai.data.preprocessing.splitting import SplitName, chronological_split
from aegis_ai.evaluation.anomaly import (
    compute_binary_metrics,
    compute_detection_delay,
    compute_grouped_detection_delay,
    positive_segments,
)
from aegis_ai.features.normalization import IdentityTimeSeriesScaler, TimeSeriesStandardScaler
from aegis_ai.features.timeseries import (
    FEATURE_COLUMNS,
    CausalRollingFeatureConfig,
    build_causal_rolling_features,
    feature_matrix,
    finite_feature_mask,
)
from aegis_ai.features.windowing import (
    SequenceWindowConfig,
    SequenceWindowSet,
    make_sequence_windows,
)
from aegis_ai.ml.anomaly.autoencoder import DenseAutoencoderConfig, DenseAutoencoderDetector
from aegis_ai.ml.anomaly.baselines import (
    RollingZScoreBaseline,
    RollingZScoreBaselineConfig,
    quantile_threshold,
)
from aegis_ai.ml.anomaly.isolation_forest import IsolationForestConfig, IsolationForestDetector
from aegis_ai.ml.anomaly.lstm_autoencoder import LSTMAutoencoderConfig, LSTMAutoencoderDetector
from aegis_ai.ml.anomaly.nab_experiment import NabSeries, labels_for_series, load_processed_nab
from aegis_ai.ml.anomaly.nab_labels import audit_nab_label_alignment
from aegis_ai.ml.anomaly.torch_training import process_rss_mb, set_reproducible_seed

matplotlib.rcParams.update(
    {
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "font.size": 10,
    }
)

ModelName = Literal[
    "rolling_zscore",
    "isolation_forest",
    "dense_autoencoder",
    "lstm_autoencoder",
]
FeatureSet = Literal["raw", "causal"]
NormalizationMode = Literal["standard", "none"]
ThresholdStrategy = Literal["train_percentile", "validation_percentile"]

MODEL_NAMES: tuple[ModelName, ...] = (
    "rolling_zscore",
    "isolation_forest",
    "dense_autoencoder",
    "lstm_autoencoder",
)


@dataclass(frozen=True)
class ThresholdConfig:
    """Threshold-selection policy."""

    strategy: ThresholdStrategy = "train_percentile"
    percentile: float = 99.0

    def __post_init__(self) -> None:
        if self.strategy not in {"train_percentile", "validation_percentile"}:
            raise ValueError(f"unsupported threshold strategy: {self.strategy}")
        if not 0 < self.percentile < 100:
            raise ValueError("percentile must be in (0, 100)")


@dataclass(frozen=True)
class NabDeepDataConfig:
    """Data protocol choices for deep anomaly experiments."""

    feature_set: FeatureSet = "raw"
    normalization: NormalizationMode = "standard"
    exclude_train_anomalies: bool = True

    def __post_init__(self) -> None:
        if self.feature_set not in {"raw", "causal"}:
            raise ValueError(f"unsupported feature_set: {self.feature_set}")
        if self.normalization not in {"standard", "none"}:
            raise ValueError(f"unsupported normalization mode: {self.normalization}")


@dataclass(frozen=True)
class AblationVariantConfig:
    """One focused ablation variant."""

    name: str
    sequence_length: int | None = None
    normalization: NormalizationMode | None = None
    feature_set: FeatureSet | None = None
    threshold_strategy: ThresholdStrategy | None = None
    threshold_percentile: float | None = None


@dataclass(frozen=True)
class AblationConfig:
    """Small ablation study configuration."""

    enabled: bool = True
    max_series: int = 6
    dense_epochs: int = 2
    variants: tuple[AblationVariantConfig, ...] = (
        AblationVariantConfig(name="main_protocol"),
        AblationVariantConfig(name="window_16", sequence_length=16),
        AblationVariantConfig(name="no_normalization", normalization="none"),
        AblationVariantConfig(name="causal_features", feature_set="causal"),
        AblationVariantConfig(
            name="validation_threshold", threshold_strategy="validation_percentile"
        ),
    )

    def __post_init__(self) -> None:
        if self.max_series < 1:
            raise ValueError("ablation max_series must be positive")
        if self.dense_epochs < 1:
            raise ValueError("ablation dense_epochs must be positive")


@dataclass(frozen=True)
class NabDeepExperimentConfig:
    """Configuration for the Phase 4 NAB deep-learning experiment."""

    train_fraction: float = 0.5
    validation_fraction: float = 0.2
    random_seed: int = 42
    max_series: int | None = None
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    data: NabDeepDataConfig = field(default_factory=NabDeepDataConfig)
    windowing: SequenceWindowConfig = field(default_factory=SequenceWindowConfig)
    feature_config: CausalRollingFeatureConfig = field(default_factory=CausalRollingFeatureConfig)
    zscore_config: RollingZScoreBaselineConfig = field(default_factory=RollingZScoreBaselineConfig)
    isolation_forest_config: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    dense_autoencoder_config: DenseAutoencoderConfig = field(default_factory=DenseAutoencoderConfig)
    lstm_autoencoder_config: LSTMAutoencoderConfig = field(default_factory=LSTMAutoencoderConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test split")
        if self.max_series is not None and self.max_series < 1:
            raise ValueError("max_series must be positive when provided")


@dataclass(frozen=True)
class PreparedSeries:
    """Split-local features, labels, and windows for one series."""

    series: NabSeries
    full_frame: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    test_eval: pd.DataFrame
    train_windows: SequenceWindowSet
    validation_windows: SequenceWindowSet
    test_windows: SequenceWindowSet
    train_fit_windows: np.ndarray
    train_threshold_frame: pd.DataFrame
    train_fit_frame: pd.DataFrame
    scaler_payload: dict[str, Any]
    deep_feature_columns: tuple[str, ...]
    train_fit_window_count: int
    excluded_train_window_count: int
    excluded_train_row_count: int


@dataclass(frozen=True)
class ModelOutcome:
    """Scores and efficiency for one model on one series."""

    model_name: ModelName
    threshold: float
    scores: np.ndarray
    predictions: np.ndarray
    training_seconds: float
    inference_seconds: float
    parameter_count: int | None
    peak_rss_mb: float | None
    train_rows: int
    validation_rows: int
    test_rows: int


@dataclass(frozen=True)
class NabDeepExperimentResult:
    """Location and summary for a completed Phase 4 run."""

    run_dir: Path
    metrics: dict[str, Any]
    dataset_summary: dict[str, Any]
    label_alignment: dict[str, Any]


def load_nab_deep_config(path: Path) -> NabDeepExperimentConfig:
    """Load a Phase 4 experiment config from YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a YAML mapping")
    return NabDeepExperimentConfig(
        train_fraction=float(payload.get("train_fraction", 0.5)),
        validation_fraction=float(payload.get("validation_fraction", 0.2)),
        random_seed=int(payload.get("random_seed", 42)),
        max_series=(int(payload["max_series"]) if payload.get("max_series") is not None else None),
        threshold=ThresholdConfig(**_mapping(payload.get("threshold"))),
        data=NabDeepDataConfig(**_mapping(payload.get("data"))),
        windowing=SequenceWindowConfig(**_mapping(payload.get("windowing"))),
        feature_config=CausalRollingFeatureConfig(**_mapping(payload.get("feature_config"))),
        zscore_config=RollingZScoreBaselineConfig(**_mapping(payload.get("zscore_config"))),
        isolation_forest_config=IsolationForestConfig(
            **_mapping(payload.get("isolation_forest_config"))
        ),
        dense_autoencoder_config=DenseAutoencoderConfig.from_dict(
            _mapping(payload.get("dense_autoencoder_config"))
        ),
        lstm_autoencoder_config=LSTMAutoencoderConfig.from_dict(
            _mapping(payload.get("lstm_autoencoder_config"))
        ),
        ablation=_load_ablation_config(_mapping(payload.get("ablation"))),
    )


def run_nab_deep_anomaly_experiment(
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    config: NabDeepExperimentConfig | None = None,
    config_path: Path | None = None,
) -> NabDeepExperimentResult:
    """Run Phase 4 NAB anomaly detection with baselines and autoencoders."""

    root = root or project_root()
    cfg = config or (
        load_nab_deep_config(config_path) if config_path is not None else NabDeepExperimentConfig()
    )
    set_reproducible_seed(cfg.random_seed)
    run_dir = _create_run_dir(
        output_root or root / "experiments" / "anomaly" / "nab" / "deep_autoencoders",
        run_id,
    )
    for directory in ["figures", "models", "checkpoints", "metadata"]:
        (run_dir / directory).mkdir(parents=True, exist_ok=True)

    all_series = load_processed_nab(root)
    label_alignment = audit_nab_label_alignment(all_series)
    eval_series = _select_eval_series(all_series, cfg.max_series)

    per_series_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    efficiency_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []

    for series in eval_series:
        prepared = _prepare_series(series, cfg)
        outcomes = _evaluate_all_models(prepared, cfg, run_dir, save_models=True)
        prediction_frame = prepared.test_eval[
            ["entity_id", "timestamp", "metric_name", "metric_value", "y_true"]
        ].copy()
        for outcome in outcomes:
            prediction_frame[f"{outcome.model_name}_score"] = outcome.scores
            prediction_frame[f"{outcome.model_name}_prediction"] = outcome.predictions
            prediction_frame[f"{outcome.model_name}_threshold"] = outcome.threshold
            per_series_rows.append(_series_metric_row(prepared, outcome))
            efficiency_rows.append(_efficiency_row(prepared, outcome))
        prediction_frames.append(prediction_frame)
        protocol_rows.append(_protocol_row(prepared))

    per_series = pd.DataFrame(per_series_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    efficiency = pd.DataFrame(efficiency_rows)
    protocol = pd.DataFrame(protocol_rows)
    aggregate_metrics = _aggregate_metrics(predictions, per_series, efficiency)
    error_analysis = _build_error_analysis(predictions)
    dataset_summary = _dataset_summary(all_series, eval_series, predictions, cfg)
    ablation_results = (
        _run_ablation_study(eval_series, cfg, run_dir) if cfg.ablation.enabled else {}
    )

    _write_json(run_dir / "config.json", _config_payload(cfg))
    _write_json(run_dir / "dataset_summary.json", dataset_summary)
    _write_json(run_dir / "label_alignment.json", label_alignment)
    _write_json(run_dir / "metrics.json", aggregate_metrics)
    _write_json(run_dir / "error_analysis.json", error_analysis)
    _write_json(run_dir / "ablation_results.json", ablation_results)
    per_series.to_csv(run_dir / "per_series_metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    efficiency.to_csv(run_dir / "efficiency.csv", index=False)
    protocol.to_csv(run_dir / "protocol_audit.csv", index=False)
    _plot_all(run_dir, predictions, per_series, aggregate_metrics, ablation_results)
    _write_run_report(run_dir, aggregate_metrics, dataset_summary, ablation_results)

    return NabDeepExperimentResult(
        run_dir=run_dir,
        metrics=aggregate_metrics,
        dataset_summary=dataset_summary,
        label_alignment=label_alignment,
    )


def _prepare_series(series: NabSeries, cfg: NabDeepExperimentConfig) -> PreparedSeries:
    feature_frame = build_causal_rolling_features(series.frame, config=cfg.feature_config)
    feature_frame["y_true"] = labels_for_series(series)
    feature_frame["split"] = ""
    bounds = chronological_split(
        len(feature_frame),
        train_fraction=cfg.train_fraction,
        validation_fraction=cfg.validation_fraction,
    )
    for bound in bounds:
        feature_frame.loc[bound.start : bound.end - 1, "split"] = bound.name.value

    usable = finite_feature_mask(feature_frame)
    feature_frame = feature_frame.loc[usable].copy().reset_index(drop=True)
    deep_columns = _deep_feature_columns(cfg.data.feature_set)

    train = (
        feature_frame[feature_frame["split"].eq(SplitName.TRAIN.value)]
        .copy()
        .reset_index(drop=True)
    )
    validation = (
        feature_frame[feature_frame["split"].eq(SplitName.VALIDATION.value)]
        .copy()
        .reset_index(drop=True)
    )
    test = (
        feature_frame[feature_frame["split"].eq(SplitName.TEST.value)].copy().reset_index(drop=True)
    )
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"series does not have enough split rows: {series.entity_id}")

    train_fit_frame = _normal_frame(train, exclude=cfg.data.exclude_train_anomalies)
    train_threshold_frame = train_fit_frame
    excluded_train_rows = int(len(train) - len(train_fit_frame))
    scaler = _fit_scaler(
        train_fit_frame.loc[:, list(deep_columns)].to_numpy(dtype=float),
        mode=cfg.data.normalization,
    )
    scaler_payload = scaler.to_dict()
    scaled_columns = tuple(f"deep_feature_{index}" for index in range(len(deep_columns)))
    scaled_frame = feature_frame.copy()
    scaled_values = scaler.transform(scaled_frame.loc[:, list(deep_columns)].to_numpy(dtype=float))
    for index, column in enumerate(scaled_columns):
        scaled_frame[column] = scaled_values[:, index]

    train_scaled = (
        scaled_frame[scaled_frame["split"].eq(SplitName.TRAIN.value)].copy().reset_index(drop=True)
    )
    validation_scaled = (
        scaled_frame[scaled_frame["split"].eq(SplitName.VALIDATION.value)]
        .copy()
        .reset_index(drop=True)
    )
    test_scaled = (
        scaled_frame[scaled_frame["split"].eq(SplitName.TEST.value)].copy().reset_index(drop=True)
    )

    train_windows = _windows_for_frame(train_scaled, scaled_columns, cfg.windowing)
    validation_windows = _windows_for_frame(validation_scaled, scaled_columns, cfg.windowing)
    test_windows = _windows_for_frame(test_scaled, scaled_columns, cfg.windowing)
    if train_windows.is_empty or validation_windows.is_empty or test_windows.is_empty:
        raise ValueError(f"series does not have enough rows for windows: {series.entity_id}")

    train_fit_windows = _normal_windows(
        train_windows,
        exclude=cfg.data.exclude_train_anomalies,
    )
    excluded_train_windows = int(train_windows.windows.shape[0] - train_fit_windows.shape[0])
    test_eval = test.iloc[test_windows.end_indices].copy().reset_index(drop=True)

    return PreparedSeries(
        series=series,
        full_frame=scaled_frame,
        train=train,
        validation=validation,
        test=test,
        test_eval=test_eval,
        train_windows=train_windows,
        validation_windows=validation_windows,
        test_windows=test_windows,
        train_fit_windows=train_fit_windows,
        train_threshold_frame=train_threshold_frame,
        train_fit_frame=train_fit_frame,
        scaler_payload=scaler_payload,
        deep_feature_columns=deep_columns,
        train_fit_window_count=int(train_fit_windows.shape[0]),
        excluded_train_window_count=excluded_train_windows,
        excluded_train_row_count=excluded_train_rows,
    )


def _evaluate_all_models(
    prepared: PreparedSeries,
    cfg: NabDeepExperimentConfig,
    run_dir: Path,
    *,
    save_models: bool,
) -> list[ModelOutcome]:
    return [
        _evaluate_rolling_zscore(prepared, cfg),
        _evaluate_isolation_forest(prepared, cfg, run_dir, save_model=save_models),
        _evaluate_dense_autoencoder(prepared, cfg, run_dir, save_model=save_models),
        _evaluate_lstm_autoencoder(prepared, cfg, run_dir, save_model=save_models),
    ]


def _evaluate_rolling_zscore(
    prepared: PreparedSeries,
    cfg: NabDeepExperimentConfig,
) -> ModelOutcome:
    started = time.perf_counter()
    baseline = RollingZScoreBaseline(cfg.zscore_config).fit(prepared.train_threshold_frame)
    training_seconds = float(time.perf_counter() - started)
    train_scores = baseline.score_samples(prepared.train_threshold_frame)
    validation_scores = baseline.score_samples(prepared.validation)
    threshold = _select_threshold(cfg.threshold, train_scores, validation_scores)

    started = time.perf_counter()
    scores = baseline.score_samples(prepared.test_eval)
    inference_seconds = float(time.perf_counter() - started)
    predictions = (scores >= threshold).astype(int)
    return ModelOutcome(
        model_name="rolling_zscore",
        threshold=threshold,
        scores=scores,
        predictions=predictions,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=0,
        peak_rss_mb=process_rss_mb(),
        train_rows=len(prepared.train_threshold_frame),
        validation_rows=len(prepared.validation),
        test_rows=len(prepared.test_eval),
    )


def _evaluate_isolation_forest(
    prepared: PreparedSeries,
    cfg: NabDeepExperimentConfig,
    run_dir: Path,
    *,
    save_model: bool,
) -> ModelOutcome:
    started = time.perf_counter()
    detector = IsolationForestDetector(cfg.isolation_forest_config).fit(
        feature_matrix(prepared.train_fit_frame, FEATURE_COLUMNS)
    )
    training_seconds = float(time.perf_counter() - started)
    train_scores = detector.score_samples(
        feature_matrix(prepared.train_threshold_frame, FEATURE_COLUMNS)
    )
    validation_scores = detector.score_samples(feature_matrix(prepared.validation, FEATURE_COLUMNS))
    threshold = _select_threshold(cfg.threshold, train_scores, validation_scores)

    started = time.perf_counter()
    scores = detector.score_samples(feature_matrix(prepared.test_eval, FEATURE_COLUMNS))
    inference_seconds = float(time.perf_counter() - started)
    predictions = (scores >= threshold).astype(int)
    if save_model:
        detector.save(
            run_dir
            / "models"
            / "isolation_forest"
            / f"{_safe_filename(prepared.series.entity_id)}.joblib"
        )
    return ModelOutcome(
        model_name="isolation_forest",
        threshold=threshold,
        scores=scores,
        predictions=predictions,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=None,
        peak_rss_mb=process_rss_mb(),
        train_rows=len(prepared.train_fit_frame),
        validation_rows=len(prepared.validation),
        test_rows=len(prepared.test_eval),
    )


def _evaluate_dense_autoencoder(
    prepared: PreparedSeries,
    cfg: NabDeepExperimentConfig,
    run_dir: Path,
    *,
    save_model: bool,
) -> ModelOutcome:
    detector = DenseAutoencoderDetector(cfg.dense_autoencoder_config)
    checkpoint = (
        run_dir
        / "checkpoints"
        / "dense_autoencoder"
        / f"{_safe_filename(prepared.series.entity_id)}.pt"
        if save_model
        else None
    )
    summary = detector.fit(
        prepared.train_fit_windows,
        validation_windows=prepared.validation_windows.windows,
        checkpoint_path=checkpoint,
    )
    train_scores = detector.reconstruction_error(prepared.train_fit_windows)
    validation_scores = detector.reconstruction_error(prepared.validation_windows.windows)
    threshold = _select_threshold(cfg.threshold, train_scores, validation_scores)

    started = time.perf_counter()
    scores = detector.reconstruction_error(prepared.test_windows.windows)
    inference_seconds = float(time.perf_counter() - started)
    predictions = (scores >= threshold).astype(int)
    if save_model:
        detector.save(
            run_dir
            / "models"
            / "dense_autoencoder"
            / f"{_safe_filename(prepared.series.entity_id)}.pt"
        )
    return ModelOutcome(
        model_name="dense_autoencoder",
        threshold=threshold,
        scores=scores,
        predictions=predictions,
        training_seconds=summary.training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=summary.parameter_count,
        peak_rss_mb=summary.peak_rss_mb,
        train_rows=prepared.train_fit_window_count,
        validation_rows=int(prepared.validation_windows.windows.shape[0]),
        test_rows=int(prepared.test_windows.windows.shape[0]),
    )


def _evaluate_lstm_autoencoder(
    prepared: PreparedSeries,
    cfg: NabDeepExperimentConfig,
    run_dir: Path,
    *,
    save_model: bool,
) -> ModelOutcome:
    detector = LSTMAutoencoderDetector(cfg.lstm_autoencoder_config)
    checkpoint = (
        run_dir
        / "checkpoints"
        / "lstm_autoencoder"
        / f"{_safe_filename(prepared.series.entity_id)}.pt"
        if save_model
        else None
    )
    summary = detector.fit(
        prepared.train_fit_windows,
        validation_windows=prepared.validation_windows.windows,
        checkpoint_path=checkpoint,
    )
    train_scores = detector.reconstruction_error(prepared.train_fit_windows)
    validation_scores = detector.reconstruction_error(prepared.validation_windows.windows)
    threshold = _select_threshold(cfg.threshold, train_scores, validation_scores)

    started = time.perf_counter()
    scores = detector.reconstruction_error(prepared.test_windows.windows)
    inference_seconds = float(time.perf_counter() - started)
    predictions = (scores >= threshold).astype(int)
    if save_model:
        detector.save(
            run_dir
            / "models"
            / "lstm_autoencoder"
            / f"{_safe_filename(prepared.series.entity_id)}.pt"
        )
    return ModelOutcome(
        model_name="lstm_autoencoder",
        threshold=threshold,
        scores=scores,
        predictions=predictions,
        training_seconds=summary.training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=summary.parameter_count,
        peak_rss_mb=summary.peak_rss_mb,
        train_rows=prepared.train_fit_window_count,
        validation_rows=int(prepared.validation_windows.windows.shape[0]),
        test_rows=int(prepared.test_windows.windows.shape[0]),
    )


def _run_ablation_study(
    series_items: list[NabSeries],
    cfg: NabDeepExperimentConfig,
    run_dir: Path,
) -> dict[str, Any]:
    selected_series = _select_ablation_series(series_items, cfg)
    rows: list[dict[str, Any]] = []
    per_series_rows: list[dict[str, Any]] = []

    for variant in cfg.ablation.variants:
        variant_cfg = _apply_ablation_variant(cfg, variant)
        prediction_frames: list[pd.DataFrame] = []
        efficiency_rows: list[dict[str, Any]] = []
        for series in selected_series:
            prepared = _prepare_series(series, variant_cfg)
            outcome = _evaluate_dense_autoencoder(
                prepared,
                variant_cfg,
                run_dir,
                save_model=False,
            )
            prediction_frame = prepared.test_eval[
                ["entity_id", "timestamp", "metric_name", "metric_value", "y_true"]
            ].copy()
            prediction_frame["prediction"] = outcome.predictions
            prediction_frame["score"] = outcome.scores
            prediction_frames.append(prediction_frame)
            metric_row = _series_metric_row(prepared, outcome)
            metric_row["variant"] = variant.name
            per_series_rows.append(metric_row)
            efficiency_rows.append(_efficiency_row(prepared, outcome))

        predictions = pd.concat(prediction_frames, ignore_index=True)
        y_true = predictions["y_true"].to_numpy(dtype=int)
        y_pred = predictions["prediction"].to_numpy(dtype=int)
        scores = predictions["score"].to_numpy(dtype=float)
        metrics = compute_binary_metrics(y_true, y_pred, scores).to_dict()
        delay = compute_grouped_detection_delay(
            predictions,
            y_true_column="y_true",
            y_pred_column="prediction",
        ).to_dict()
        efficiency = pd.DataFrame(efficiency_rows)
        rows.append(
            {
                "variant": variant.name,
                "model": "dense_autoencoder",
                "series_count": len(selected_series),
                "sequence_length": variant_cfg.windowing.sequence_length,
                "normalization": variant_cfg.data.normalization,
                "feature_set": variant_cfg.data.feature_set,
                "threshold_strategy": variant_cfg.threshold.strategy,
                "threshold_percentile": variant_cfg.threshold.percentile,
                **metrics,
                "detection_delay": delay,
                "training_seconds_total": _maybe_float(efficiency["training_seconds"].sum()),
                "inference_seconds_total": _maybe_float(efficiency["inference_seconds"].sum()),
            }
        )

    pd.DataFrame(rows).to_csv(run_dir / "ablation_results.csv", index=False)
    pd.DataFrame(per_series_rows).to_csv(run_dir / "ablation_per_series_metrics.csv", index=False)
    return {
        "scope": "Dense autoencoder ablation on a deterministic subset of labeled NAB series.",
        "selected_series": [series.entity_id for series in selected_series],
        "results": rows,
    }


def _series_metric_row(prepared: PreparedSeries, outcome: ModelOutcome) -> dict[str, Any]:
    y_true = prepared.test_eval["y_true"].to_numpy(dtype=int)
    metrics = compute_binary_metrics(y_true, outcome.predictions, outcome.scores).to_dict()
    delay = compute_detection_delay(
        y_true,
        outcome.predictions,
        prepared.test_eval["timestamp"].tolist(),
    ).to_dict()
    return {
        "entity_id": prepared.series.entity_id,
        "metric_name": prepared.series.metric_name,
        "model": outcome.model_name,
        "train_rows": outcome.train_rows,
        "validation_rows": outcome.validation_rows,
        "test_rows": outcome.test_rows,
        "threshold": outcome.threshold,
        **metrics,
        **{f"detection_{key}": value for key, value in delay.items()},
    }


def _efficiency_row(prepared: PreparedSeries, outcome: ModelOutcome) -> dict[str, Any]:
    return {
        "entity_id": prepared.series.entity_id,
        "model": outcome.model_name,
        "training_seconds": outcome.training_seconds,
        "inference_seconds": outcome.inference_seconds,
        "parameter_count": outcome.parameter_count,
        "peak_rss_mb": outcome.peak_rss_mb,
        "train_rows_or_windows": outcome.train_rows,
        "validation_rows_or_windows": outcome.validation_rows,
        "test_rows_or_windows": outcome.test_rows,
    }


def _protocol_row(prepared: PreparedSeries) -> dict[str, Any]:
    return {
        "entity_id": prepared.series.entity_id,
        "train_rows": len(prepared.train),
        "validation_rows": len(prepared.validation),
        "test_rows": len(prepared.test),
        "test_scored_rows": len(prepared.test_eval),
        "train_windows": int(prepared.train_windows.windows.shape[0]),
        "validation_windows": int(prepared.validation_windows.windows.shape[0]),
        "test_windows": int(prepared.test_windows.windows.shape[0]),
        "train_fit_windows": prepared.train_fit_window_count,
        "excluded_train_windows": prepared.excluded_train_window_count,
        "excluded_train_rows": prepared.excluded_train_row_count,
        "first_test_scored_timestamp": str(prepared.test_eval["timestamp"].iloc[0]),
        "last_test_scored_timestamp": str(prepared.test_eval["timestamp"].iloc[-1]),
    }


def _aggregate_metrics(
    predictions: pd.DataFrame,
    per_series: pd.DataFrame,
    efficiency: pd.DataFrame,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "global": {},
        "per_series_distribution": {},
        "best_series_by_f1": {},
        "worst_series_by_f1": {},
        "median_series_by_f1": {},
        "difficult_series": {},
        "computational_efficiency": {},
    }
    y_true = predictions["y_true"].to_numpy(dtype=int)
    for model in MODEL_NAMES:
        prediction_column = f"{model}_prediction"
        score_column = f"{model}_score"
        y_pred = predictions[prediction_column].to_numpy(dtype=int)
        scores = predictions[score_column].to_numpy(dtype=float)
        metrics = compute_binary_metrics(y_true, y_pred, scores).to_dict()
        delay = compute_grouped_detection_delay(
            predictions,
            y_true_column="y_true",
            y_pred_column=prediction_column,
        ).to_dict()
        aggregate["global"][model] = {
            **metrics,
            "detection_delay": delay,
        }
        model_series = per_series[per_series["model"].eq(model)].copy()
        aggregate["per_series_distribution"][model] = _distribution_summary(model_series)
        aggregate["best_series_by_f1"][model] = _ranked_series(model_series, best=True)
        aggregate["worst_series_by_f1"][model] = _ranked_series(model_series, best=False)
        aggregate["median_series_by_f1"][model] = _median_series(model_series)
        aggregate["difficult_series"][model] = _difficult_series(model_series)
        aggregate["computational_efficiency"][model] = _efficiency_summary(
            efficiency[efficiency["model"].eq(model)].copy()
        )
    return aggregate


def _distribution_summary(model_series: pd.DataFrame) -> dict[str, Any]:
    fields = ["precision", "recall", "f1", "pr_auc", "false_positive_rate"]
    summary: dict[str, Any] = {
        "series_evaluated": int(len(model_series)),
        "positive_series": int((model_series["positives"] > 0).sum()),
        "f1_zero_series": int((model_series["f1"] == 0).sum()),
    }
    for field_name in fields:
        values = pd.to_numeric(model_series[field_name], errors="coerce").dropna()
        summary[field_name] = {
            "mean": _maybe_float(values.mean()),
            "median": _maybe_float(values.median()),
            "std": _maybe_float(values.std(ddof=0)),
        }
    return summary


def _ranked_series(model_series: pd.DataFrame, *, best: bool) -> list[dict[str, Any]]:
    ranked = model_series[model_series["positives"] > 0].sort_values(
        ["f1", "recall", "precision"],
        ascending=not best,
    )
    fields = ["entity_id", "precision", "recall", "f1", "pr_auc", "positives", "threshold"]
    return cast(list[dict[str, Any]], ranked.loc[:, fields].head(5).to_dict(orient="records"))


def _median_series(model_series: pd.DataFrame) -> list[dict[str, Any]]:
    positive = model_series[model_series["positives"] > 0].copy()
    if positive.empty:
        return []
    median_f1 = float(positive["f1"].median())
    positive["distance_from_median_f1"] = (positive["f1"] - median_f1).abs()
    ranked = positive.sort_values(["distance_from_median_f1", "recall", "precision"])
    fields = ["entity_id", "precision", "recall", "f1", "pr_auc", "positives", "threshold"]
    return cast(list[dict[str, Any]], ranked.loc[:, fields].head(5).to_dict(orient="records"))


def _difficult_series(model_series: pd.DataFrame) -> list[dict[str, Any]]:
    difficult = model_series[(model_series["positives"] > 0) & (model_series["f1"] == 0)].copy()
    difficult = difficult.sort_values(["false_negatives", "pr_auc"], ascending=[False, True])
    fields = [
        "entity_id",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "false_negatives",
        "threshold",
    ]
    return cast(list[dict[str, Any]], difficult.loc[:, fields].head(10).to_dict(orient="records"))


def _efficiency_summary(model_efficiency: pd.DataFrame) -> dict[str, Any]:
    if model_efficiency.empty:
        return {}
    parameter_values = pd.to_numeric(model_efficiency["parameter_count"], errors="coerce").dropna()
    memory_values = pd.to_numeric(model_efficiency["peak_rss_mb"], errors="coerce").dropna()
    return {
        "training_seconds_total": _maybe_float(model_efficiency["training_seconds"].sum()),
        "training_seconds_mean": _maybe_float(model_efficiency["training_seconds"].mean()),
        "inference_seconds_total": _maybe_float(model_efficiency["inference_seconds"].sum()),
        "inference_seconds_mean": _maybe_float(model_efficiency["inference_seconds"].mean()),
        "parameter_count_median": _maybe_float(parameter_values.median()),
        "peak_rss_mb_max": _maybe_float(memory_values.max()),
    }


def _build_error_analysis(predictions: pd.DataFrame) -> dict[str, Any]:
    return {
        model: _model_error_analysis(
            predictions,
            prediction_column=f"{model}_prediction",
            score_column=f"{model}_score",
        )
        for model in MODEL_NAMES
    }


def _model_error_analysis(
    predictions: pd.DataFrame,
    *,
    prediction_column: str,
    score_column: str,
) -> dict[str, Any]:
    false_positives = predictions[
        predictions[prediction_column].eq(1) & predictions["y_true"].eq(0)
    ].sort_values(score_column, ascending=False)
    false_positive_examples = [
        _row_example(row, score_column) for _, row in false_positives.head(10).iterrows()
    ]

    missed_windows: list[dict[str, Any]] = []
    delayed_windows: list[dict[str, Any]] = []
    for entity_id, group in predictions.groupby("entity_id", sort=True):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        truth = ordered["y_true"].to_numpy(dtype=int)
        pred = ordered[prediction_column].to_numpy(dtype=int)
        for start, end in positive_segments(truth):
            window = ordered.iloc[start:end]
            hit_offsets = np.flatnonzero(pred[start:end] == 1)
            summary = {
                "entity_id": entity_id,
                "window_start": str(window["timestamp"].iloc[0]),
                "window_end": str(window["timestamp"].iloc[-1]),
                "points": int(len(window)),
                "max_score": float(window[score_column].max()),
                "min_value": float(window["metric_value"].min()),
                "max_value": float(window["metric_value"].max()),
            }
            if hit_offsets.size == 0:
                missed_windows.append(summary)
            else:
                delay_steps = int(hit_offsets[0])
                if delay_steps > 0:
                    delayed = dict(summary)
                    delayed["delay_steps"] = delay_steps
                    delayed["first_detection"] = str(window["timestamp"].iloc[delay_steps])
                    delayed_windows.append(delayed)

    delayed_windows.sort(key=lambda item: int(item["delay_steps"]), reverse=True)
    missed_windows.sort(key=lambda item: float(item["max_score"]), reverse=True)
    return {
        "false_positive_examples": false_positive_examples,
        "missed_window_examples": missed_windows[:10],
        "delayed_window_examples": delayed_windows[:10],
    }


def _dataset_summary(
    all_series: list[NabSeries],
    eval_series: list[NabSeries],
    predictions: pd.DataFrame,
    cfg: NabDeepExperimentConfig,
) -> dict[str, Any]:
    return {
        "dataset_id": "nab",
        "series_count": len(all_series),
        "evaluated_series_count": len(eval_series),
        "observation_count": int(sum(len(series.frame) for series in all_series)),
        "evaluated_test_rows": int(len(predictions)),
        "test_positive_points": int(predictions["y_true"].sum()),
        "split": {
            "strategy": "per-series chronological split",
            "train_fraction": cfg.train_fraction,
            "validation_fraction": cfg.validation_fraction,
            "test_fraction": 1 - cfg.train_fraction - cfg.validation_fraction,
        },
        "windowing": asdict(cfg.windowing),
        "normalization": {
            "mode": cfg.data.normalization,
            "fit_scope": (
                "training rows excluding labeled anomalies"
                if cfg.data.exclude_train_anomalies
                else "all training rows"
            ),
        },
        "threshold": asdict(cfg.threshold),
        "deep_feature_set": cfg.data.feature_set,
        "score_alignment": (
            "Autoencoder window scores are assigned to each window's ending timestamp. "
            "Baselines are evaluated on the same window-end test timestamps."
        ),
    }


def _plot_all(
    run_dir: Path,
    predictions: pd.DataFrame,
    per_series: pd.DataFrame,
    aggregate_metrics: dict[str, Any],
    ablation_results: dict[str, Any],
) -> None:
    example_entity = _choose_example_entity(per_series, model="lstm_autoencoder")
    example = predictions[predictions["entity_id"].eq(example_entity)].sort_values("timestamp")
    _plot_ground_truth(run_dir, example)
    _plot_isolation_forest_predictions(run_dir, example)
    _plot_autoencoder_scores(run_dir, example, model="dense_autoencoder")
    _plot_autoencoder_scores(run_dir, example, model="lstm_autoencoder")
    _plot_model_comparison(run_dir, example)
    _plot_reconstruction_distributions(run_dir, predictions)
    _plot_per_series_f1_distribution(run_dir, per_series)
    _plot_ablation_results(run_dir, ablation_results)
    _plot_metric_comparison(run_dir, aggregate_metrics)


def _plot_ground_truth(run_dir: Path, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame["timestamp"], frame["metric_value"], color="#1f4e79", linewidth=1.1)
    positives = frame[frame["y_true"].eq(1)]
    if not positives.empty:
        ax.scatter(
            positives["timestamp"],
            positives["metric_value"],
            s=13,
            color="#f59e0b",
            label="Ground truth interval points",
            zorder=3,
        )
    ax.set_title(f"NAB Ground Truth: {frame['entity_id'].iloc[0]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Metric value")
    _legend_if_labels(ax, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "01_original_ground_truth_intervals.png")
    plt.close(fig)


def _plot_isolation_forest_predictions(run_dir: Path, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame["timestamp"], frame["metric_value"], color="#334155", linewidth=1.0)
    positives = frame[frame["y_true"].eq(1)]
    detections = frame[frame["isolation_forest_prediction"].eq(1)]
    ax.scatter(
        positives["timestamp"],
        positives["metric_value"],
        s=13,
        color="#f59e0b",
        label="Ground truth",
    )
    ax.scatter(
        detections["timestamp"],
        detections["metric_value"],
        s=20,
        marker="o",
        facecolors="none",
        edgecolors="#dc2626",
        label="Isolation Forest",
    )
    ax.set_title(f"Isolation Forest Predictions: {frame['entity_id'].iloc[0]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Metric value")
    _legend_if_labels(ax, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "02_isolation_forest_predictions.png")
    plt.close(fig)


def _plot_autoencoder_scores(
    run_dir: Path,
    frame: pd.DataFrame,
    *,
    model: Literal["dense_autoencoder", "lstm_autoencoder"],
) -> None:
    title = "Dense Autoencoder" if model == "dense_autoencoder" else "LSTM Autoencoder"
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame["timestamp"], frame[f"{model}_score"], color="#2563eb", linewidth=1.0)
    threshold = float(frame[f"{model}_threshold"].iloc[0])
    ax.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.0, label="Threshold")
    positives = frame[frame["y_true"].eq(1)]
    if not positives.empty:
        ax.scatter(
            positives["timestamp"],
            positives[f"{model}_score"],
            s=14,
            color="#f59e0b",
            label="Ground truth",
        )
    ax.set_title(f"{title} Scores: {frame['entity_id'].iloc[0]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Reconstruction error")
    _legend_if_labels(ax, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    filename = (
        "03_dense_autoencoder_scores.png"
        if model == "dense_autoencoder"
        else "04_lstm_autoencoder_scores.png"
    )
    fig.savefig(run_dir / "figures" / filename)
    plt.close(fig)


def _plot_model_comparison(run_dir: Path, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame["timestamp"], frame["metric_value"], color="#334155", linewidth=1.0)
    colors = {
        "rolling_zscore": "#7c3aed",
        "isolation_forest": "#dc2626",
        "dense_autoencoder": "#0891b2",
        "lstm_autoencoder": "#16a34a",
    }
    for model, color in colors.items():
        hits = frame[frame[f"{model}_prediction"].eq(1)]
        ax.scatter(
            hits["timestamp"],
            hits["metric_value"],
            s=14,
            label=model,
            color=color,
            alpha=0.75,
        )
    positives = frame[frame["y_true"].eq(1)]
    ax.scatter(
        positives["timestamp"],
        positives["metric_value"],
        s=12,
        color="#f59e0b",
        label="ground_truth",
        alpha=0.55,
    )
    ax.set_title(f"Model Comparison: {frame['entity_id'].iloc[0]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Metric value")
    _legend_if_labels(ax, loc="best", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "05_model_comparison_same_series.png")
    plt.close(fig)


def _plot_reconstruction_distributions(run_dir: Path, predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, color in {
        "dense_autoencoder": "#0891b2",
        "lstm_autoencoder": "#16a34a",
    }.items():
        values = predictions[f"{model}_score"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        ax.hist(finite, bins=60, alpha=0.55, label=model, color=color)
    ax.set_title("Reconstruction Error Distribution")
    ax.set_xlabel("Reconstruction error")
    ax.set_ylabel("Count")
    _legend_if_labels(ax, loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "06_reconstruction_error_distribution.png")
    plt.close(fig)


def _plot_per_series_f1_distribution(run_dir: Path, per_series: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, color in {
        "rolling_zscore": "#7c3aed",
        "isolation_forest": "#dc2626",
        "dense_autoencoder": "#0891b2",
        "lstm_autoencoder": "#16a34a",
    }.items():
        values = per_series.loc[per_series["model"].eq(model), "f1"].to_numpy(dtype=float)
        ax.hist(values, bins=25, alpha=0.45, label=model, color=color)
    ax.set_title("Per-Series F1 Distribution")
    ax.set_xlabel("F1")
    ax.set_ylabel("Series count")
    _legend_if_labels(ax, loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "07_per_series_f1_distribution.png")
    plt.close(fig)


def _plot_ablation_results(run_dir: Path, ablation_results: dict[str, Any]) -> None:
    rows = ablation_results.get("results", [])
    if not rows:
        return
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(frame["variant"], frame["f1"], color="#2563eb")
    ax.set_title("Dense Autoencoder Ablation F1")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Global F1 on ablation subset")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "08_dense_autoencoder_ablation_f1.png")
    plt.close(fig)


def _plot_metric_comparison(run_dir: Path, aggregate_metrics: dict[str, Any]) -> None:
    fields = ["precision", "recall", "f1", "pr_auc"]
    rows = []
    for model in MODEL_NAMES:
        rows.append([aggregate_metrics["global"][model][field] for field in fields])
    values = np.asarray(rows, dtype=float)
    x = np.arange(len(fields))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#7c3aed", "#dc2626", "#0891b2", "#16a34a"]
    for index, model in enumerate(MODEL_NAMES):
        ax.bar(x + (index - 1.5) * width, values[index], width, label=model, color=colors[index])
    ax.set_ylim(0, max(1.0, float(np.nanmax(values)) * 1.2))
    ax.set_xticks(x)
    ax.set_xticklabels(
        [field.upper() if field == "f1" else field.replace("_", "-") for field in fields]
    )
    ax.set_title("Global Test Metrics")
    _legend_if_labels(ax, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "09_model_metric_comparison.png")
    plt.close(fig)


def _legend_if_labels(ax: Any, **kwargs: Any) -> None:
    _handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(**kwargs)


def _write_run_report(
    run_dir: Path,
    metrics: dict[str, Any],
    dataset_summary: dict[str, Any],
    ablation_results: dict[str, Any],
) -> None:
    rows = []
    for model in MODEL_NAMES:
        values = metrics["global"][model]
        delay = values["detection_delay"]
        rows.append(
            "| {model} | {precision:.4f} | {recall:.4f} | {f1:.4f} | "
            "{pr_auc:.4f} | {fpr:.4f} | {missed} |".format(
                model=model,
                precision=values["precision"],
                recall=values["recall"],
                f1=values["f1"],
                pr_auc=values["pr_auc"] or 0.0,
                fpr=values["false_positive_rate"],
                missed=delay["missed_windows"],
            )
        )
    ablation_note = "Ablation disabled."
    if ablation_results.get("results"):
        best = max(ablation_results["results"], key=lambda row: float(row["f1"]))
        ablation_note = f"Best dense ablation by F1: `{best['variant']}` ({best['f1']:.4f})."

    report = f"""# NAB Deep Anomaly Detection Run

Run directory: `{run_dir.as_posix()}`

## Dataset

- Evaluated series: {dataset_summary["evaluated_series_count"]}
- Total NAB series: {dataset_summary["series_count"]}
- Test rows evaluated: {dataset_summary["evaluated_test_rows"]:,}
- Test positive points: {dataset_summary["test_positive_points"]:,}
- Window size: {dataset_summary["windowing"]["sequence_length"]}
- Window stride: {dataset_summary["windowing"]["stride"]}

## Global Metrics

| Model | Precision | Recall | F1 | PR-AUC | FPR | Missed Windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Ablation

{ablation_note}
"""
    (run_dir / "run_report.md").write_text(report, encoding="utf-8")


def _select_eval_series(series_items: list[NabSeries], max_series: int | None) -> list[NabSeries]:
    if max_series is None:
        return series_items
    return series_items[:max_series]


def _select_ablation_series(
    series_items: list[NabSeries],
    cfg: NabDeepExperimentConfig,
) -> list[NabSeries]:
    candidates: list[NabSeries] = []
    for series in sorted(series_items, key=lambda item: item.entity_id):
        feature_frame = build_causal_rolling_features(series.frame, config=cfg.feature_config)
        feature_frame["y_true"] = labels_for_series(series)
        bounds = chronological_split(
            len(feature_frame),
            train_fraction=cfg.train_fraction,
            validation_fraction=cfg.validation_fraction,
        )
        test_bound = bounds[-1]
        test_labels = feature_frame["y_true"].iloc[test_bound.start : test_bound.end]
        if int(test_labels.sum()) > 0:
            candidates.append(series)
        if len(candidates) >= cfg.ablation.max_series:
            break
    if len(candidates) < cfg.ablation.max_series:
        for series in series_items:
            if series not in candidates:
                candidates.append(series)
            if len(candidates) >= cfg.ablation.max_series:
                break
    return candidates


def _apply_ablation_variant(
    cfg: NabDeepExperimentConfig,
    variant: AblationVariantConfig,
) -> NabDeepExperimentConfig:
    windowing = cfg.windowing
    if variant.sequence_length is not None:
        windowing = replace(windowing, sequence_length=variant.sequence_length)
    data = cfg.data
    if variant.normalization is not None:
        data = replace(data, normalization=variant.normalization)
    if variant.feature_set is not None:
        data = replace(data, feature_set=variant.feature_set)
    threshold = cfg.threshold
    if variant.threshold_strategy is not None:
        threshold = replace(threshold, strategy=variant.threshold_strategy)
    if variant.threshold_percentile is not None:
        threshold = replace(threshold, percentile=variant.threshold_percentile)
    dense_config = replace(cfg.dense_autoencoder_config, epochs=cfg.ablation.dense_epochs)
    return replace(
        cfg,
        windowing=windowing,
        data=data,
        threshold=threshold,
        dense_autoencoder_config=dense_config,
        ablation=replace(cfg.ablation, enabled=False),
    )


def _deep_feature_columns(feature_set: FeatureSet) -> tuple[str, ...]:
    if feature_set == "raw":
        return ("raw_value",)
    return FEATURE_COLUMNS


def _fit_scaler(
    values: np.ndarray,
    *,
    mode: NormalizationMode,
) -> TimeSeriesStandardScaler | IdentityTimeSeriesScaler:
    if mode == "none":
        return IdentityTimeSeriesScaler().fit(values)
    return TimeSeriesStandardScaler().fit(values)


def _normal_frame(frame: pd.DataFrame, *, exclude: bool) -> pd.DataFrame:
    if not exclude:
        return frame
    normal = frame[frame["y_true"].eq(0)].copy()
    return normal if not normal.empty else frame


def _normal_windows(windows: SequenceWindowSet, *, exclude: bool) -> np.ndarray:
    if not exclude or windows.labels is None:
        return windows.windows
    normal = windows.windows[windows.labels == 0]
    return cast(np.ndarray, normal if normal.shape[0] else windows.windows)


def _windows_for_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    config: SequenceWindowConfig,
) -> SequenceWindowSet:
    return make_sequence_windows(
        frame.loc[:, list(feature_columns)].to_numpy(dtype=float),
        timestamps=frame["timestamp"].tolist(),
        labels=frame["y_true"].to_numpy(dtype=int),
        config=config,
    )


def _select_threshold(
    config: ThresholdConfig,
    train_scores: np.ndarray,
    validation_scores: np.ndarray,
) -> float:
    source = validation_scores if config.strategy == "validation_percentile" else train_scores
    if source.shape[0] == 0:
        source = train_scores
    return quantile_threshold(source, config.percentile / 100)


def _row_example(row: pd.Series, score_column: str) -> dict[str, Any]:
    return {
        "entity_id": row["entity_id"],
        "timestamp": str(row["timestamp"]),
        "metric_value": float(row["metric_value"]),
        "score": float(row[score_column]),
    }


def _choose_example_entity(per_series: pd.DataFrame, *, model: ModelName) -> str:
    candidates = per_series[(per_series["model"].eq(model)) & (per_series["positives"] > 0)].copy()
    if candidates.empty:
        return str(per_series["entity_id"].iloc[0])
    candidates = candidates.sort_values(["f1", "recall", "precision"], ascending=False)
    return str(candidates["entity_id"].iloc[0])


def _create_run_dir(output_root: Path, run_id: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = run_id or f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_root / resolved
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{resolved}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )


def _config_payload(cfg: NabDeepExperimentConfig) -> dict[str, Any]:
    return {
        "train_fraction": cfg.train_fraction,
        "validation_fraction": cfg.validation_fraction,
        "random_seed": cfg.random_seed,
        "max_series": cfg.max_series,
        "threshold": asdict(cfg.threshold),
        "data": asdict(cfg.data),
        "windowing": asdict(cfg.windowing),
        "feature_config": asdict(cfg.feature_config),
        "zscore_config": asdict(cfg.zscore_config),
        "isolation_forest_config": asdict(cfg.isolation_forest_config),
        "dense_autoencoder_config": asdict(cfg.dense_autoencoder_config),
        "lstm_autoencoder_config": asdict(cfg.lstm_autoencoder_config),
        "ablation": {
            "enabled": cfg.ablation.enabled,
            "max_series": cfg.ablation.max_series,
            "dense_epochs": cfg.ablation.dense_epochs,
            "variants": [asdict(variant) for variant in cfg.ablation.variants],
        },
    }


def _load_ablation_config(payload: dict[str, Any]) -> AblationConfig:
    if not payload:
        return AblationConfig()
    values = dict(payload)
    raw_variants = values.pop("variants", None)
    variants = (
        tuple(AblationVariantConfig(**variant) for variant in raw_variants)
        if raw_variants is not None
        else AblationConfig().variants
    )
    return AblationConfig(variants=variants, **values)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("configuration section must be a mapping")
    return cast(dict[str, Any], value)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in value)
    return safe.strip("_") or "series"


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
