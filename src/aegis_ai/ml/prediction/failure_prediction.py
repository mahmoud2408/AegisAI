"""Phase 7 supervised failure prediction and predictive-maintenance experiments."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aegis_ai.data.adapters.ai4i import FAILURE_COLUMNS as AI4I_FAILURE_COLUMNS
from aegis_ai.data.adapters.ai4i import NUMERIC_FEATURES as AI4I_NUMERIC_FEATURES
from aegis_ai.data.adapters.metropt import BINARY_STATE_COLUMNS, CONTINUOUS_SENSORS
from aegis_ai.data.dataset_registry import project_root

_XGBClassifier: Any | None
try:  # pragma: no cover - availability depends on optional ml extra.
    from xgboost import XGBClassifier as _ImportedXGBClassifier

    _XGBClassifier = _ImportedXGBClassifier
except Exception:  # pragma: no cover
    _XGBClassifier = None

_shap: Any | None
try:  # pragma: no cover - availability depends on optional ml extra.
    import shap as _imported_shap

    _shap = _imported_shap
except Exception:  # pragma: no cover
    _shap = None


matplotlib.rcParams.update(
    {
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "font.size": 9,
    }
)

ModelName = Literal["logistic_regression", "random_forest", "xgboost"]
DatasetName = Literal["ai4i", "metropt"]
SplitName = Literal["train", "validation", "test"]

AI4I_CATEGORICAL_FEATURES = ("Type",)
AI4I_IDENTIFIER_COLUMNS = ("UDI", "Product ID")
DEFAULT_THRESHOLD_GRID = (0.1, 0.2, 0.3, 0.5, 0.7)


class SupportsPredictProba(Protocol):
    """Protocol for fitted classifiers used in this experiment."""

    def predict_proba(self, x: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class FailureEvent:
    """Curated failure event interval."""

    event_id: str
    start_time: str
    end_time: str
    failure_type: str
    severity: str
    source: str
    report: str = ""

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.start_time)

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.end_time)


@dataclass(frozen=True)
class AI4IExperimentConfig:
    """AI4I supervised baseline configuration."""

    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID

    def __post_init__(self) -> None:
        _validate_split_fractions(self.train_fraction, self.validation_fraction)
        _validate_threshold_grid(self.threshold_grid)


@dataclass(frozen=True)
class MetroPTExperimentConfig:
    """MetroPT temporal predictive-maintenance configuration."""

    horizon_hours: float = 6.0
    prediction_stride_rows: int = 6
    train_end: str = "2020-05-31 00:00:00"
    validation_end: str = "2020-06-08 00:00:00"
    lag_steps: tuple[int, ...] = (1, 6, 60)
    rolling_windows: tuple[str, ...] = ("5min", "30min")
    trend_window: str = "30min"
    min_periods: int = 3
    max_train_rows: int = 60000
    negative_to_positive_ratio: int = 25
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID
    failure_events: tuple[FailureEvent, ...] = (
        FailureEvent(
            event_id="metropt_air_leak_2020_04_18",
            start_time="2020-04-18 00:00:00",
            end_time="2020-04-18 23:59:00",
            failure_type="air_leak",
            severity="high_stress",
            source="data/raw/metropt/Data Description_Metro.pdf",
        ),
        FailureEvent(
            event_id="metropt_air_leak_2020_05_29",
            start_time="2020-05-29 23:30:00",
            end_time="2020-05-30 06:00:00",
            failure_type="air_leak",
            severity="high_stress",
            source="data/raw/metropt/Data Description_Metro.pdf",
            report="Maintenance report text is present in the source PDF table.",
        ),
        FailureEvent(
            event_id="metropt_air_leak_2020_06_05",
            start_time="2020-06-05 10:00:00",
            end_time="2020-06-07 14:30:00",
            failure_type="air_leak",
            severity="high_stress",
            source="data/raw/metropt/Data Description_Metro.pdf",
            report="Maintenance on 8 Jun at 16:00 in the source PDF table.",
        ),
        FailureEvent(
            event_id="metropt_air_leak_2020_07_15",
            start_time="2020-07-15 14:30:00",
            end_time="2020-07-15 19:00:00",
            failure_type="air_leak",
            severity="high_stress",
            source="data/raw/metropt/Data Description_Metro.pdf",
            report="Maintenance on 16 Jul at 00:00 in the source PDF table.",
        ),
    )

    def __post_init__(self) -> None:
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        if self.prediction_stride_rows < 1:
            raise ValueError("prediction_stride_rows must be positive")
        if not self.lag_steps or any(step < 1 for step in self.lag_steps):
            raise ValueError("lag_steps must contain positive integers")
        if not self.rolling_windows:
            raise ValueError("rolling_windows must not be empty")
        if self.min_periods < 1:
            raise ValueError("min_periods must be positive")
        if self.max_train_rows < 1:
            raise ValueError("max_train_rows must be positive")
        if self.negative_to_positive_ratio < 1:
            raise ValueError("negative_to_positive_ratio must be positive")
        _validate_threshold_grid(self.threshold_grid)
        if pd.Timestamp(self.train_end) >= pd.Timestamp(self.validation_end):
            raise ValueError("train_end must be before validation_end")


@dataclass(frozen=True)
class FailureModelConfig:
    """Shared classical model configuration."""

    random_seed: int = 42
    logistic_max_iter: int = 1000
    random_forest_estimators: int = 200
    random_forest_min_samples_leaf: int = 3
    xgboost_estimators: int = 200
    xgboost_max_depth: int = 4
    xgboost_learning_rate: float = 0.05
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.logistic_max_iter < 1:
            raise ValueError("logistic_max_iter must be positive")
        if self.random_forest_estimators < 1 or self.xgboost_estimators < 1:
            raise ValueError("estimator counts must be positive")
        if self.random_forest_min_samples_leaf < 1:
            raise ValueError("random_forest_min_samples_leaf must be positive")
        if self.xgboost_max_depth < 1:
            raise ValueError("xgboost_max_depth must be positive")
        if self.xgboost_learning_rate <= 0:
            raise ValueError("xgboost_learning_rate must be positive")


@dataclass(frozen=True)
class SHAPConfig:
    """SHAP artifact configuration."""

    enabled: bool = True
    sample_rows: int = 200
    local_explanation_rows: int = 5

    def __post_init__(self) -> None:
        if self.sample_rows < 1:
            raise ValueError("sample_rows must be positive")
        if self.local_explanation_rows < 1:
            raise ValueError("local_explanation_rows must be positive")


@dataclass(frozen=True)
class FailurePredictionExperimentConfig:
    """Phase 7 experiment configuration."""

    random_seed: int = 42
    ai4i: AI4IExperimentConfig = field(default_factory=AI4IExperimentConfig)
    metropt: MetroPTExperimentConfig = field(default_factory=MetroPTExperimentConfig)
    models: FailureModelConfig = field(default_factory=FailureModelConfig)
    shap: SHAPConfig = field(default_factory=SHAPConfig)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Binary classification metrics for failure prediction."""

    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float | None
    false_positive_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    support: int
    positives: int
    predicted_positives: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdSelection:
    """Validation-selected probability threshold."""

    threshold: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    validation_false_positive_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeadTimeSummary:
    """Early-warning summary for temporal failure events."""

    failure_events: int
    warned_events: int
    missed_events: int
    warning_coverage: float
    mean_lead_time_hours: float | None
    median_lead_time_hours: float | None
    false_alarm_rate: float
    false_alarms_per_day: float
    false_warnings: int
    warning_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetResult:
    """Experiment result for one dataset."""

    output_dir: Path
    metrics: dict[str, Any]
    audit: dict[str, Any]


@dataclass(frozen=True)
class FailurePredictionExperimentResult:
    """Location and summary for a completed Phase 7 run."""

    run_dir: Path
    metrics: dict[str, Any]


def load_failure_prediction_config(path: Path) -> FailurePredictionExperimentConfig:
    """Load Phase 7 failure-prediction config from YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a YAML mapping")
    return FailurePredictionExperimentConfig(
        random_seed=int(payload.get("random_seed", 42)),
        ai4i=_load_ai4i_config(_mapping(payload.get("ai4i"))),
        metropt=_load_metropt_config(_mapping(payload.get("metropt"))),
        models=FailureModelConfig(**_mapping(payload.get("models"))),
        shap=SHAPConfig(**_mapping(payload.get("shap"))),
    )


def run_failure_prediction_experiment(
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    config: FailurePredictionExperimentConfig | None = None,
) -> FailurePredictionExperimentResult:
    """Run AI4I and MetroPT Phase 7 failure-prediction experiments."""

    cfg = config or FailurePredictionExperimentConfig()
    project = project_root() if root is None else root
    resolved_run_id = run_id or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    base_output = output_root or project / "experiments" / "prediction" / "failure_prediction"
    run_dir = base_output / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_json(run_dir / "config.json", asdict(cfg))

    started = time.perf_counter()
    ai4i_result = run_ai4i_failure_prediction(project, run_dir / "ai4i", cfg)
    metropt_result = run_metropt_failure_prediction(project, run_dir / "metropt", cfg)
    metrics = {
        "run_id": resolved_run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "ai4i": ai4i_result.metrics,
        "metropt": metropt_result.metrics,
        "best_models": {
            "ai4i": ai4i_result.metrics["best_model"],
            "metropt": metropt_result.metrics["best_model"],
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _save_json(run_dir / "metrics.json", metrics)
    _write_phase_report(run_dir, ai4i_result=ai4i_result, metropt_result=metropt_result)
    return FailurePredictionExperimentResult(run_dir=run_dir, metrics=metrics)


def run_ai4i_failure_prediction(
    root: Path,
    output_dir: Path,
    config: FailurePredictionExperimentConfig,
) -> DatasetResult:
    """Run non-temporal supervised failure prediction on AI4I."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    shap_dir = output_dir / "shap"
    for directory in (figures_dir, models_dir, shap_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame = load_ai4i_frame(root)
    audit = audit_ai4i_frame(frame)
    _save_json(output_dir / "dataset_audit.json", audit)

    feature_columns = tuple(AI4I_NUMERIC_FEATURES) + AI4I_CATEGORICAL_FEATURES
    y = frame["Machine failure"].astype(int)
    x = frame.loc[:, list(feature_columns)]
    split = stratified_ai4i_split(
        y.to_numpy(dtype=int),
        train_fraction=config.ai4i.train_fraction,
        validation_fraction=config.ai4i.validation_fraction,
        random_seed=config.random_seed,
    )
    positive_weight = _positive_weight(y.iloc[split["train"]].to_numpy(dtype=int))
    preprocessor = _tabular_preprocessor(
        numeric_features=tuple(AI4I_NUMERIC_FEATURES),
        categorical_features=AI4I_CATEGORICAL_FEATURES,
    )
    model_results = _run_model_family(
        dataset="ai4i",
        x_train=x.iloc[split["train"]],
        y_train=y.iloc[split["train"]].to_numpy(dtype=int),
        x_validation=x.iloc[split["validation"]],
        y_validation=y.iloc[split["validation"]].to_numpy(dtype=int),
        x_test=x.iloc[split["test"]],
        y_test=y.iloc[split["test"]].to_numpy(dtype=int),
        preprocessor=preprocessor,
        model_config=config.models,
        threshold_grid=config.ai4i.threshold_grid,
        positive_weight=positive_weight,
        models_dir=models_dir,
    )
    metrics_df = pd.DataFrame([result["test_metrics"] for result in model_results])
    thresholds_df = pd.DataFrame([result["threshold_selection"] for result in model_results])
    predictions_df = pd.concat(
        [cast(pd.DataFrame, result["predictions"]) for result in model_results],
        ignore_index=True,
    )
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    thresholds_df.to_csv(output_dir / "thresholds.csv", index=False)
    _write_dataframe(predictions_df, output_dir / "predictions")

    final_tree = _select_final_tree_model(model_results)
    shap_status = _write_shap_artifacts(
        final_tree,
        x_reference=x.iloc[split["test"]],
        output_dir=shap_dir,
        config=config.shap,
    )
    _plot_ai4i_model_comparison(metrics_df, figures_dir)
    _plot_calibration(model_results, figures_dir, dataset="ai4i")
    _plot_feature_importance(final_tree, figures_dir / "final_tree_feature_importance.png")

    best = _best_metrics_record(metrics_df)
    metrics = {
        "task": "binary_machine_failure_prediction",
        "target_definition": "Machine failure == 1",
        "split_policy": (
            "Stratified train/validation/test split because AI4I has row index only "
            "and no meaningful temporal axis."
        ),
        "feature_columns": list(feature_columns),
        "identifier_columns_excluded": list(AI4I_IDENTIFIER_COLUMNS),
        "failure_mode_columns_excluded_from_features": list(AI4I_FAILURE_COLUMNS),
        "class_balance": audit["class_balance"],
        "best_model": best,
        "models": _records_by_key(metrics_df, key="model"),
        "thresholds": _records_by_key(thresholds_df, key="model"),
        "shap": shap_status,
    }
    _save_json(output_dir / "metrics.json", metrics)
    return DatasetResult(output_dir=output_dir, metrics=metrics, audit=audit)


def run_metropt_failure_prediction(
    root: Path,
    output_dir: Path,
    config: FailurePredictionExperimentConfig,
) -> DatasetResult:
    """Run temporal predictive-maintenance baseline on MetroPT."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    shap_dir = output_dir / "shap"
    for directory in (figures_dir, models_dir, shap_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = load_metropt_frame(root)
    audit = audit_metropt_frame(raw, failure_events=config.metropt.failure_events)
    labeled = build_metropt_failure_target(
        raw,
        failure_events=config.metropt.failure_events,
        horizon_hours=config.metropt.horizon_hours,
    )
    features = build_metropt_causal_features(
        labeled,
        continuous_columns=CONTINUOUS_SENSORS,
        state_columns=BINARY_STATE_COLUMNS,
        lag_steps=config.metropt.lag_steps,
        rolling_windows=config.metropt.rolling_windows,
        trend_window=config.metropt.trend_window,
        min_periods=config.metropt.min_periods,
    )
    modeling = prepare_metropt_modeling_frame(features, config.metropt)
    feature_columns = tuple(
        column for column in modeling.columns if column not in _METROPT_NON_FEATURE_COLUMNS
    )
    preprocessing_metadata = {
        "horizon_hours": config.metropt.horizon_hours,
        "prediction_stride_rows": config.metropt.prediction_stride_rows,
        "lag_steps": list(config.metropt.lag_steps),
        "rolling_windows": list(config.metropt.rolling_windows),
        "trend_window": config.metropt.trend_window,
        "excluded_active_failure_rows": int(labeled["is_failure_period"].sum()),
        "feature_rows_after_dropna_and_stride": int(modeling.shape[0]),
        "feature_columns": list(feature_columns),
    }
    _save_json(output_dir / "dataset_audit.json", audit)
    _save_json(output_dir / "preprocessing_metadata.json", preprocessing_metadata)

    split_counts = modeling.groupby("split", observed=True)["target"].agg(["count", "sum"])
    split_counts.to_csv(output_dir / "split_counts.csv")
    train_frame = modeling[modeling["split"].eq("train")]
    validation_frame = modeling[modeling["split"].eq("validation")]
    test_frame = modeling[modeling["split"].eq("test")]
    train_sample = sample_temporal_training_rows(
        train_frame,
        target_column="target",
        max_rows=config.metropt.max_train_rows,
        negative_to_positive_ratio=config.metropt.negative_to_positive_ratio,
    )
    positive_weight = _positive_weight(train_sample["target"].to_numpy(dtype=int))
    preprocessor = _numeric_preprocessor(feature_columns)
    model_results = _run_model_family(
        dataset="metropt",
        x_train=train_sample.loc[:, list(feature_columns)],
        y_train=train_sample["target"].to_numpy(dtype=int),
        x_validation=validation_frame.loc[:, list(feature_columns)],
        y_validation=validation_frame["target"].to_numpy(dtype=int),
        x_test=test_frame.loc[:, list(feature_columns)],
        y_test=test_frame["target"].to_numpy(dtype=int),
        preprocessor=preprocessor,
        model_config=config.models,
        threshold_grid=config.metropt.threshold_grid,
        positive_weight=positive_weight,
        models_dir=models_dir,
        test_metadata=test_frame.loc[:, ["timestamp", "sequence_index", "target"]],
    )
    metrics_rows: list[dict[str, Any]] = []
    thresholds_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    episode_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    test_events = _events_in_split(
        config.metropt.failure_events,
        start=pd.Timestamp(config.metropt.validation_end),
        end=raw["timestamp"].max(),
    )
    for result in model_results:
        predictions = cast(pd.DataFrame, result["predictions"])
        lead_summary, episode_df = compute_early_warning_metrics(
            predictions,
            failure_events=test_events,
            horizon_hours=config.metropt.horizon_hours,
            threshold=float(result["threshold_selection"]["threshold"]),
        )
        metrics = dict(cast(dict[str, Any], result["test_metrics"]))
        metrics.update(
            {
                "warning_coverage": lead_summary.warning_coverage,
                "median_lead_time_hours": lead_summary.median_lead_time_hours,
                "false_alarm_rate": lead_summary.false_alarm_rate,
                "false_alarms_per_day": lead_summary.false_alarms_per_day,
                "missed_failure_events": lead_summary.missed_events,
            }
        )
        metrics_rows.append(metrics)
        thresholds_rows.append(cast(dict[str, Any], result["threshold_selection"]))
        lead_row = lead_summary.to_dict()
        lead_row["model"] = result["model"]
        lead_rows.append(lead_row)
        episode_df["model"] = result["model"]
        episode_frames.append(episode_df)
        prediction_frames.append(predictions)

    metrics_df = pd.DataFrame(metrics_rows)
    thresholds_df = pd.DataFrame(thresholds_rows)
    lead_df = pd.DataFrame(lead_rows)
    episodes_df = pd.concat(episode_frames, ignore_index=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    thresholds_df.to_csv(output_dir / "thresholds.csv", index=False)
    lead_df.to_csv(output_dir / "lead_time_summary.csv", index=False)
    episodes_df.to_csv(output_dir / "failure_episode_analysis.csv", index=False)
    _write_dataframe(predictions_df, output_dir / "predictions")

    final_tree = _select_final_tree_model(model_results)
    shap_status = _write_shap_artifacts(
        final_tree,
        x_reference=test_frame.loc[:, list(feature_columns)],
        output_dir=shap_dir,
        config=config.shap,
    )
    _plot_metropt_timeline(
        modeling,
        predictions_df,
        events=config.metropt.failure_events,
        figures_dir=figures_dir,
        best_model=str(_best_metrics_record(metrics_df)["model"]),
    )
    _plot_lead_time(episodes_df, figures_dir)
    _plot_metropt_model_comparison(metrics_df, figures_dir)
    _plot_feature_importance(final_tree, figures_dir / "final_tree_feature_importance.png")

    best = _best_metrics_record(metrics_df)
    metrics = {
        "task": "temporal_failure_within_horizon_prediction",
        "target_definition": (
            f"failure_within_{config.metropt.horizon_hours:g}h: at time t, target=1 "
            "when a curated MetroPT failure start occurs in (t, t+horizon]. "
            "Rows during active failure periods are excluded from supervised training "
            "and evaluation."
        ),
        "label_source": "Curated from data/raw/metropt/Data Description_Metro.pdf",
        "split_policy": (
            "Event-aware chronological split: first two failures in train, third in "
            "validation, fourth in test. Boundaries avoid splitting active failure "
            "periods."
        ),
        "split_counts": split_counts.reset_index().to_dict(orient="records"),
        "best_model": best,
        "models": _records_by_key(metrics_df, key="model"),
        "thresholds": _records_by_key(thresholds_df, key="model"),
        "lead_time": _records_by_key(lead_df, key="model"),
        "shap": shap_status,
    }
    _save_json(output_dir / "metrics.json", metrics)
    return DatasetResult(output_dir=output_dir, metrics=metrics, audit=audit)


def load_ai4i_frame(root: Path) -> pd.DataFrame:
    """Load the raw AI4I CSV."""

    path = root / "data" / "raw" / "ai4i" / "ai4i2020.csv"
    if not path.exists():
        raise FileNotFoundError(f"AI4I CSV not found: {path}")
    return pd.read_csv(path)


def audit_ai4i_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Audit AI4I labels, feature types, and integrity."""

    required = (
        set(AI4I_IDENTIFIER_COLUMNS)
        | {"Type"}
        | set(AI4I_NUMERIC_FEATURES)
        | set(AI4I_FAILURE_COLUMNS)
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"AI4I frame missing required columns: {missing}")
    target = frame["Machine failure"].astype(int)
    modes = {
        column: frame[column].astype(int).value_counts().sort_index().to_dict()
        for column in AI4I_FAILURE_COLUMNS
    }
    return {
        "dataset_id": "ai4i",
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "target_column": "Machine failure",
        "target_definition": "Machine failure == 1",
        "identifier_columns": list(AI4I_IDENTIFIER_COLUMNS),
        "numeric_features": list(AI4I_NUMERIC_FEATURES),
        "categorical_features": list(AI4I_CATEGORICAL_FEATURES),
        "failure_columns": list(AI4I_FAILURE_COLUMNS),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "class_balance": {
            "negative": int((target == 0).sum()),
            "positive": int((target == 1).sum()),
            "positive_rate": float(target.mean()),
        },
        "type_counts": frame["Type"].value_counts().sort_index().to_dict(),
        "failure_label_counts": modes,
        "positive_failure_mode_counts": {
            column: int(frame.loc[target.eq(1), column].astype(int).sum())
            for column in AI4I_FAILURE_COLUMNS
            if column != "Machine failure"
        },
        "numeric_describe": frame.loc[:, list(AI4I_NUMERIC_FEATURES)].describe().to_dict(),
        "temporal_axis": "none; row index is not a meaningful time axis",
    }


def stratified_ai4i_split(
    y: np.ndarray,
    *,
    train_fraction: float,
    validation_fraction: float,
    random_seed: int,
) -> dict[SplitName, np.ndarray]:
    """Create a deterministic stratified AI4I split."""

    _validate_split_fractions(train_fraction, validation_fraction)
    indices = np.arange(y.shape[0], dtype=int)
    test_fraction = 1 - train_fraction - validation_fraction
    temp_fraction = validation_fraction + test_fraction
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        y,
        test_size=temp_fraction,
        random_state=random_seed,
        stratify=y,
    )
    relative_test_fraction = test_fraction / temp_fraction
    validation_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_fraction,
        random_state=random_seed,
        stratify=y_temp,
    )
    return {
        "train": np.sort(train_idx),
        "validation": np.sort(validation_idx),
        "test": np.sort(test_idx),
    }


def load_metropt_frame(root: Path) -> pd.DataFrame:
    """Load MetroPT raw wide telemetry."""

    path = root / "data" / "raw" / "metropt" / "MetroPT3(AirCompressor).csv"
    if not path.exists():
        raise FileNotFoundError(f"MetroPT CSV not found: {path}")
    frame = pd.read_csv(path)
    if "Unnamed: 0" in frame.columns:
        frame = frame.rename(columns={"Unnamed: 0": "unnamed_index"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame.sort_values("timestamp").reset_index(drop=True)


def audit_metropt_frame(
    frame: pd.DataFrame,
    *,
    failure_events: tuple[FailureEvent, ...],
) -> dict[str, Any]:
    """Audit MetroPT telemetry and curated failure events."""

    sensor_columns = tuple(CONTINUOUS_SENSORS) + tuple(BINARY_STATE_COLUMNS)
    missing = sorted(set(sensor_columns).union({"timestamp"}).difference(frame.columns))
    if missing:
        raise ValueError(f"MetroPT frame missing required columns: {missing}")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    deltas = timestamps.diff().dropna().dt.total_seconds()
    state_values = {
        column: sorted(float(value) for value in frame[column].dropna().unique().tolist())
        for column in BINARY_STATE_COLUMNS
    }
    failure_rows = _active_failure_mask(timestamps, failure_events)
    return {
        "dataset_id": "metropt",
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "timestamp_start": timestamps.min().isoformat(),
        "timestamp_end": timestamps.max().isoformat(),
        "continuous_sensors": list(CONTINUOUS_SENSORS),
        "binary_state_columns": list(BINARY_STATE_COLUMNS),
        "machine_readable_failure_label_column": False,
        "failure_event_source": "data/raw/metropt/Data Description_Metro.pdf",
        "curated_failure_events": [asdict(event) for event in failure_events],
        "curated_failure_event_count": len(failure_events),
        "rows_in_curated_failure_periods": int(failure_rows.sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "median_cadence_seconds": float(deltas.median()),
        "dominant_cadence_seconds": {
            str(key): int(value) for key, value in deltas.value_counts().head(10).items()
        },
        "gaps_gt_1_5x_median": int((deltas > 1.5 * float(deltas.median())).sum()),
        "constant_features": [
            column for column in sensor_columns if frame[column].nunique(dropna=False) <= 1
        ],
        "binary_state_values": state_values,
        "numeric_describe": frame.loc[:, list(CONTINUOUS_SENSORS)].describe().to_dict(),
    }


def build_metropt_failure_target(
    frame: pd.DataFrame,
    *,
    failure_events: tuple[FailureEvent, ...],
    horizon_hours: float,
) -> pd.DataFrame:
    """Add a causal future-failure target from curated MetroPT event starts."""

    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    result = frame.sort_values("timestamp").reset_index(drop=True).copy()
    timestamps = pd.to_datetime(result["timestamp"], errors="raise")
    horizon = pd.Timedelta(hours=horizon_hours)
    target = np.zeros(result.shape[0], dtype=np.int8)
    event_id = np.full(result.shape[0], "", dtype=object)
    for event in failure_events:
        mask = (timestamps < event.start) & (timestamps >= event.start - horizon)
        target[mask.to_numpy()] = 1
        event_id[mask.to_numpy()] = event.event_id
    active = _active_failure_mask(timestamps, failure_events)
    result["target"] = target
    result["prediction_target_name"] = f"failure_within_{horizon_hours:g}h"
    result["future_failure_event_id"] = event_id
    result["is_failure_period"] = active.astype(bool)
    result["eligible_for_prediction"] = ~result["is_failure_period"]
    return result


def build_metropt_causal_features(
    frame: pd.DataFrame,
    *,
    continuous_columns: tuple[str, ...],
    state_columns: tuple[str, ...],
    lag_steps: tuple[int, ...],
    rolling_windows: tuple[str, ...],
    trend_window: str,
    min_periods: int,
    state_transition_count_window: str | None = None,
) -> pd.DataFrame:
    """Create MetroPT features using only current and historical observations."""

    ordered = frame.sort_values("timestamp").reset_index(drop=True).copy()
    timestamps = pd.to_datetime(ordered["timestamp"], errors="raise")
    result_columns: dict[str, Any] = {
        "timestamp": ordered["timestamp"].to_numpy(),
        "target": ordered["target"].to_numpy(dtype=np.int8),
        "future_failure_event_id": ordered["future_failure_event_id"].to_numpy(dtype=object),
        "is_failure_period": ordered["is_failure_period"].to_numpy(dtype=bool),
        "eligible_for_prediction": ordered["eligible_for_prediction"].to_numpy(dtype=bool),
        "sequence_index": np.arange(ordered.shape[0], dtype=np.int64),
    }
    indexed = ordered.set_index("timestamp", drop=False)

    for column in continuous_columns:
        values = pd.to_numeric(indexed[column], errors="coerce").astype(float)
        result_columns[f"{column}__current"] = values.to_numpy(dtype=np.float32)
        for step in lag_steps:
            result_columns[f"{column}__lag_{step}"] = values.shift(step).to_numpy(dtype=np.float32)
        diff_1 = values - values.shift(1)
        result_columns[f"{column}__diff_1"] = diff_1.to_numpy(dtype=np.float32)
        denominator = values.shift(1).where(values.shift(1).abs() > 1e-8)
        result_columns[f"{column}__rate_1"] = (
            (diff_1 / denominator)
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .to_numpy(dtype=np.float32)
        )
        for window in rolling_windows:
            rolling = values.rolling(window, min_periods=min_periods)
            suffix = _window_suffix(window)
            result_columns[f"{column}__rolling_mean_{suffix}"] = rolling.mean().to_numpy(
                dtype=np.float32
            )
            result_columns[f"{column}__rolling_std_{suffix}"] = rolling.std(ddof=0).to_numpy(
                dtype=np.float32
            )
            result_columns[f"{column}__rolling_min_{suffix}"] = rolling.min().to_numpy(
                dtype=np.float32
            )
            result_columns[f"{column}__rolling_max_{suffix}"] = rolling.max().to_numpy(
                dtype=np.float32
            )
        result_columns[f"{column}__trend_{_window_suffix(trend_window)}"] = _causal_window_rate(
            values=values,
            timestamps=timestamps,
            window=pd.Timedelta(trend_window),
            min_periods=min_periods,
        )

    for column in state_columns:
        values = pd.to_numeric(ordered[column], errors="coerce").astype(float)
        transitions = values.ne(values.shift(1)).astype(int)
        transitions.iloc[0] = 0
        transition_times = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        transition_times.loc[transitions.eq(1)] = timestamps.loc[transitions.eq(1)]
        last_transition = transition_times.ffill()
        seconds_since = (timestamps - last_transition).dt.total_seconds().fillna(0.0)
        result_columns[f"{column}__current"] = values.to_numpy(dtype=np.float32)
        result_columns[f"{column}__lag_1"] = values.shift(1).to_numpy(dtype=np.float32)
        result_columns[f"{column}__transition"] = transitions.to_numpy(dtype=np.float32)
        result_columns[f"{column}__seconds_since_transition"] = seconds_since.to_numpy(
            dtype=np.float32
        )
        if state_transition_count_window is not None:
            transition_series = pd.Series(
                transitions.to_numpy(dtype=float),
                index=pd.DatetimeIndex(timestamps),
            )
            result_columns[
                f"{column}__transition_count_{_window_suffix(state_transition_count_window)}"
            ] = (
                transition_series.rolling(
                    state_transition_count_window,
                    min_periods=min_periods,
                )
                .sum()
                .to_numpy(dtype=np.float32)
            )

    hour = timestamps.dt.hour + timestamps.dt.minute / 60 + timestamps.dt.second / 3600
    day = timestamps.dt.dayofweek.astype(float)
    result_columns["time__hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    result_columns["time__hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    result_columns["time__dayofweek_sin"] = np.sin(2 * np.pi * day / 7).astype(np.float32)
    result_columns["time__dayofweek_cos"] = np.cos(2 * np.pi * day / 7).astype(np.float32)
    return pd.DataFrame(result_columns)


_METROPT_NON_FEATURE_COLUMNS = {
    "timestamp",
    "target",
    "future_failure_event_id",
    "is_failure_period",
    "eligible_for_prediction",
    "sequence_index",
    "split",
    "prediction_target_name",
}


def prepare_metropt_modeling_frame(
    features: pd.DataFrame,
    config: MetroPTExperimentConfig,
) -> pd.DataFrame:
    """Filter MetroPT feature rows, apply stride, and assign chronological splits."""

    feature_columns = [
        column for column in features.columns if column not in _METROPT_NON_FEATURE_COLUMNS
    ]
    frame = features.copy()
    finite_mask = np.isfinite(frame.loc[:, feature_columns].to_numpy(dtype=float)).all(axis=1)
    stride_mask = frame["sequence_index"].mod(config.prediction_stride_rows).eq(0)
    eligible = frame["eligible_for_prediction"].astype(bool)
    frame = frame.loc[finite_mask & stride_mask & eligible].copy()
    train_end = pd.Timestamp(config.train_end)
    validation_end = pd.Timestamp(config.validation_end)
    frame["split"] = np.select(
        [
            frame["timestamp"] < train_end,
            (frame["timestamp"] >= train_end) & (frame["timestamp"] < validation_end),
        ],
        ["train", "validation"],
        default="test",
    )
    split_positive_counts = frame.groupby("split", observed=True)["target"].sum().to_dict()
    for split in ("train", "validation", "test"):
        if split not in set(frame["split"].unique().tolist()):
            raise ValueError(f"MetroPT split has no rows: {split}")
        if int(split_positive_counts.get(split, 0)) == 0:
            raise ValueError(f"MetroPT split has no positive target rows: {split}")
    return frame.reset_index(drop=True)


def sample_temporal_training_rows(
    frame: pd.DataFrame,
    *,
    target_column: str,
    max_rows: int,
    negative_to_positive_ratio: int,
) -> pd.DataFrame:
    """Keep all positive rows and evenly sample train negatives in chronological order."""

    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    positives = frame[frame[target_column].eq(1)]
    negatives = frame[frame[target_column].eq(0)]
    if positives.empty:
        return _sample_evenly(frame, max_rows)
    negative_limit = min(
        int(negatives.shape[0]),
        max(max_rows - int(positives.shape[0]), 0),
        int(positives.shape[0]) * negative_to_positive_ratio,
    )
    sampled_negatives = _sample_evenly(negatives, negative_limit)
    sampled = pd.concat([positives, sampled_negatives], axis=0).sort_values("timestamp")
    if sampled.shape[0] > max_rows:
        sampled = _sample_evenly(sampled, max_rows)
    return sampled.reset_index(drop=True)


def optimize_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...],
) -> ThresholdSelection:
    """Select a decision threshold from validation data only."""

    best: ThresholdSelection | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        metrics = classification_metrics(y_true, predictions, probabilities)
        key = (
            metrics.f1,
            metrics.recall,
            -metrics.false_positive_rate,
            -threshold,
        )
        if best is None or best_key is None or key > best_key:
            best = ThresholdSelection(
                threshold=float(threshold),
                validation_precision=metrics.precision,
                validation_recall=metrics.recall,
                validation_f1=metrics.f1,
                validation_false_positive_rate=metrics.false_positive_rate,
            )
            best_key = key
    if best is None:
        raise ValueError("threshold grid must not be empty")
    return best


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> ClassificationMetrics:
    """Compute binary classification metrics without relying on accuracy."""

    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    prob = np.asarray(probabilities, dtype=float)
    if truth.shape != pred.shape or truth.shape != prob.shape:
        raise ValueError("truth, predictions, and probabilities must have matching shapes")
    if truth.ndim != 1:
        raise ValueError("classification inputs must be 1D")
    if truth.size == 0:
        raise ValueError("cannot evaluate empty predictions")
    labels = np.asarray([0, 1], dtype=int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=labels).ravel()
    roc_auc = float(roc_auc_score(truth, prob)) if len(set(truth.tolist())) == 2 else None
    pr_auc = float(average_precision_score(truth, prob)) if len(set(truth.tolist())) == 2 else None
    brier = float(brier_score_loss(truth, prob)) if np.isfinite(prob).all() else None
    return ClassificationMetrics(
        precision=float(precision_score(truth, pred, zero_division=0)),
        recall=float(recall_score(truth, pred, zero_division=0)),
        f1=float(f1_score(truth, pred, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        false_positive_rate=float(fp / (fp + tn)) if (fp + tn) else 0.0,
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        support=int(truth.size),
        positives=int((truth == 1).sum()),
        predicted_positives=int((pred == 1).sum()),
    )


def compute_early_warning_metrics(
    predictions: pd.DataFrame,
    *,
    failure_events: tuple[FailureEvent, ...],
    horizon_hours: float,
    threshold: float,
) -> tuple[LeadTimeSummary, pd.DataFrame]:
    """Measure event-level warning lead time and false alarms."""

    if not {"timestamp", "probability", "target"}.issubset(predictions.columns):
        raise ValueError("predictions must include timestamp, probability, and target")
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["warning"] = frame["probability"].astype(float) >= threshold
    horizon = pd.Timedelta(hours=horizon_hours)
    episode_rows: list[dict[str, Any]] = []
    protected_warning_indices: set[int] = set()
    lead_times: list[float] = []
    for event in failure_events:
        window = frame[
            (frame["timestamp"] < event.start) & (frame["timestamp"] >= event.start - horizon)
        ]
        warning_window = window[window["warning"]]
        first_warning_time = None
        lead_hours = None
        if not warning_window.empty:
            first_warning_time = pd.Timestamp(warning_window["timestamp"].min())
            lead_hours = float((event.start - first_warning_time).total_seconds() / 3600)
            lead_times.append(lead_hours)
            protected_warning_indices.update(warning_window.index.astype(int).tolist())
        episode_rows.append(
            {
                "event_id": event.event_id,
                "failure_start": event.start.isoformat(),
                "failure_end": event.end.isoformat(),
                "failure_type": event.failure_type,
                "severity": event.severity,
                "warning_produced": not warning_window.empty,
                "first_warning_time": (
                    None if first_warning_time is None else first_warning_time.isoformat()
                ),
                "lead_time_hours": lead_hours,
                "warnings_in_horizon": int(warning_window.shape[0]),
                "prediction_rows_in_horizon": int(window.shape[0]),
                "max_probability_in_horizon": (
                    float(window["probability"].max()) if not window.empty else None
                ),
            }
        )
    warning_indices = set(frame.index[frame["warning"]].astype(int).tolist())
    false_warnings = len(warning_indices.difference(protected_warning_indices))
    negative_rows = int((frame["target"].astype(int) == 0).sum())
    duration_days = _duration_days(frame["timestamp"])
    summary = LeadTimeSummary(
        failure_events=len(failure_events),
        warned_events=len(lead_times),
        missed_events=len(failure_events) - len(lead_times),
        warning_coverage=float(len(lead_times) / len(failure_events)) if failure_events else 0.0,
        mean_lead_time_hours=float(np.mean(lead_times)) if lead_times else None,
        median_lead_time_hours=float(np.median(lead_times)) if lead_times else None,
        false_alarm_rate=float(false_warnings / negative_rows) if negative_rows else 0.0,
        false_alarms_per_day=float(false_warnings / duration_days) if duration_days > 0 else 0.0,
        false_warnings=false_warnings,
        warning_threshold=threshold,
    )
    return summary, pd.DataFrame(episode_rows)


def _run_model_family(
    *,
    dataset: DatasetName,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    preprocessor: ColumnTransformer,
    model_config: FailureModelConfig,
    threshold_grid: tuple[float, ...],
    positive_weight: float,
    models_dir: Path,
    test_metadata: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    models = _build_models(model_config, positive_weight=positive_weight)
    results: list[dict[str, Any]] = []
    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        training_seconds = float(time.perf_counter() - started)

        started = time.perf_counter()
        validation_prob = _positive_probabilities(pipeline, x_validation)
        test_prob = _positive_probabilities(pipeline, x_test)
        inference_seconds = float(time.perf_counter() - started)

        threshold = optimize_threshold(y_validation, validation_prob, threshold_grid)
        test_pred = (test_prob >= threshold.threshold).astype(int)
        test_metrics = classification_metrics(y_test, test_pred, test_prob).to_dict()
        test_metrics.update(
            {
                "dataset": dataset,
                "model": model_name,
                "threshold": threshold.threshold,
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "train_rows": int(y_train.shape[0]),
                "validation_rows": int(y_validation.shape[0]),
                "test_rows": int(y_test.shape[0]),
            }
        )
        threshold_record = threshold.to_dict()
        threshold_record.update({"dataset": dataset, "model": model_name})
        predictions = pd.DataFrame(
            {
                "dataset": dataset,
                "model": model_name,
                "row_number": np.arange(y_test.shape[0], dtype=int),
                "y_true": y_test.astype(np.int8),
                "probability": test_prob.astype(np.float32),
                "threshold": np.full(y_test.shape[0], threshold.threshold, dtype=np.float32),
                "y_pred": test_pred.astype(np.int8),
            }
        )
        if test_metadata is not None:
            predictions = pd.concat(
                [test_metadata.reset_index(drop=True), predictions],
                axis=1,
            )
        joblib.dump(pipeline, models_dir / f"{model_name}.joblib")
        _write_feature_importance_table(
            pipeline,
            models_dir / f"{model_name}_feature_importance.csv",
        )
        results.append(
            {
                "dataset": dataset,
                "model": model_name,
                "pipeline": pipeline,
                "validation_probabilities": validation_prob,
                "test_probabilities": test_prob,
                "test_metrics": test_metrics,
                "threshold_selection": threshold_record,
                "predictions": predictions,
                "validation_f1": threshold.validation_f1,
            }
        )
    return results


def _build_models(
    config: FailureModelConfig,
    *,
    positive_weight: float,
) -> dict[ModelName, BaseEstimator]:
    models: dict[ModelName, BaseEstimator] = {
        "logistic_regression": LogisticRegression(
            max_iter=config.logistic_max_iter,
            class_weight="balanced",
            random_state=config.random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config.random_forest_estimators,
            min_samples_leaf=config.random_forest_min_samples_leaf,
            class_weight="balanced",
            random_state=config.random_seed,
            n_jobs=config.n_jobs,
        ),
    }
    if _XGBClassifier is not None:
        models["xgboost"] = _XGBClassifier(
            n_estimators=config.xgboost_estimators,
            max_depth=config.xgboost_max_depth,
            learning_rate=config.xgboost_learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=positive_weight,
            random_state=config.random_seed,
            n_jobs=config.n_jobs,
            verbosity=0,
        )
    return models


def _tabular_preprocessor(
    *,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(numeric_features)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_features),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _numeric_preprocessor(feature_columns: tuple[str, ...]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("numeric", StandardScaler(), list(feature_columns))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _positive_probabilities(model: SupportsPredictProba, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("classifier must expose binary predict_proba output")
    return probabilities[:, 1].astype(float)


def _select_final_tree_model(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    tree_results = [
        result for result in model_results if result["model"] in {"random_forest", "xgboost"}
    ]
    if not tree_results:
        raise ValueError("no tree-based model result available")
    return max(
        tree_results,
        key=lambda result: (
            float(result["validation_f1"]),
            float(result["threshold_selection"]["validation_precision"]),
            float(result["threshold_selection"]["validation_recall"]),
            -float(result["threshold_selection"]["validation_false_positive_rate"]),
        ),
    )


def _write_shap_artifacts(
    model_result: dict[str, Any],
    *,
    x_reference: pd.DataFrame,
    output_dir: Path,
    config: SHAPConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any]
    if not config.enabled:
        status = {"status": "disabled"}
        _save_json(output_dir / "status.json", status)
        return status
    if _shap is None:
        status = {"status": "skipped", "reason": "shap is not installed"}
        _save_json(output_dir / "status.json", status)
        return status
    try:
        pipeline = cast(Pipeline, model_result["pipeline"])
        preprocessor = cast(ColumnTransformer, pipeline.named_steps["preprocessor"])
        estimator = pipeline.named_steps["model"]
        sample = _sample_evenly(x_reference, config.sample_rows)
        transformed = preprocessor.transform(sample)
        feature_names = list(preprocessor.get_feature_names_out())
        explainer = _shap.TreeExplainer(estimator)
        raw_values = explainer.shap_values(transformed)
        shap_values = _positive_class_shap_values(raw_values)
        global_importance = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        global_importance.to_csv(output_dir / "global_feature_importance.csv", index=False)

        sample_probabilities = _positive_probabilities(pipeline, sample)
        top_rows = np.argsort(-sample_probabilities)[: config.local_explanation_rows]
        local_records: list[dict[str, Any]] = []
        for local_rank, row_index in enumerate(top_rows, start=1):
            feature_order = np.argsort(-np.abs(shap_values[row_index]))[:10]
            for feature_rank, feature_index in enumerate(feature_order, start=1):
                local_records.append(
                    {
                        "local_rank": local_rank,
                        "sample_row_position": int(row_index),
                        "probability": float(sample_probabilities[row_index]),
                        "feature_rank": feature_rank,
                        "feature": feature_names[int(feature_index)],
                        "shap_value": float(shap_values[row_index, feature_index]),
                        "abs_shap_value": float(abs(shap_values[row_index, feature_index])),
                    }
                )
        pd.DataFrame(local_records).to_csv(output_dir / "local_explanations.csv", index=False)
        _plot_shap_global_importance(
            global_importance,
            output_dir / "shap_summary_bar.png",
            title=f"SHAP importance: {model_result['model']}",
        )
        status = {
            "status": "created",
            "model": model_result["model"],
            "sample_rows": int(sample.shape[0]),
            "global_importance_path": "global_feature_importance.csv",
            "local_explanations_path": "local_explanations.csv",
            "summary_plot_path": "shap_summary_bar.png",
            "note": "SHAP values are model-contributing features, not causal explanations.",
        }
    except Exception as exc:  # pragma: no cover - defensive artifact fallback.
        status = {
            "status": "failed",
            "model": model_result.get("model"),
            "reason": str(exc),
        }
    _save_json(output_dir / "status.json", status)
    return status


def _positive_class_shap_values(raw_values: Any) -> np.ndarray:
    if isinstance(raw_values, list):
        values = raw_values[1] if len(raw_values) > 1 else raw_values[0]
    else:
        values = raw_values
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        array = array[:, :, 1] if array.shape[2] > 1 else array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"unexpected SHAP value shape: {array.shape}")
    return array


def _write_feature_importance_table(pipeline: Pipeline, path: Path) -> None:
    preprocessor = cast(ColumnTransformer, pipeline.named_steps["preprocessor"])
    estimator = pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        importance_kind = "feature_importances_"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
        importance_kind = "abs_coefficient"
    else:
        return
    frame = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .assign(importance_kind=importance_kind)
        .sort_values("importance", ascending=False)
    )
    frame.to_csv(path, index=False)


def _plot_feature_importance(model_result: dict[str, Any], output_path: Path) -> None:
    pipeline = cast(Pipeline, model_result["pipeline"])
    preprocessor = cast(ColumnTransformer, pipeline.named_steps["preprocessor"])
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return
    frame = (
        pd.DataFrame(
            {
                "feature": list(preprocessor.get_feature_names_out()),
                "importance": np.asarray(estimator.feature_importances_, dtype=float),
            }
        )
        .sort_values("importance", ascending=False)
        .head(20)
    )
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.barh(frame["feature"][::-1], frame["importance"][::-1], color="#33658a")
    axis.set_title(f"Feature importance: {model_result['model']}")
    axis.set_xlabel("importance")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def _plot_shap_global_importance(frame: pd.DataFrame, output_path: Path, *, title: str) -> None:
    top = frame.head(20)
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#5b8c5a")
    axis.set_title(title)
    axis.set_xlabel("mean absolute SHAP value")
    figure.savefig(output_path)
    plt.close(figure)


def _plot_ai4i_model_comparison(metrics: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    ordered = metrics.sort_values("f1", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.barh(ordered["model"], ordered["f1"], color="#2f855a")
    axis.set_title("AI4I failure prediction F1")
    axis.set_xlabel("F1")
    figure.savefig(figures_dir / "model_comparison_f1.png")
    plt.close(figure)


def _plot_metropt_model_comparison(metrics: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    ordered = metrics.sort_values("f1", ascending=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].barh(ordered["model"], ordered["f1"], color="#2f855a")
    axes[0].set_title("MetroPT failure prediction F1")
    axes[0].set_xlabel("F1")
    axes[1].barh(ordered["model"], ordered["warning_coverage"], color="#2454a6")
    axes[1].set_title("Warning coverage")
    axes[1].set_xlabel("coverage")
    figure.savefig(figures_dir / "model_comparison_warning.png")
    plt.close(figure)


def _plot_calibration(
    model_results: list[dict[str, Any]],
    figures_dir: Path,
    *,
    dataset: DatasetName,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="#444444", linestyle="--", label="perfect")
    for result in model_results:
        predictions = cast(pd.DataFrame, result["predictions"])
        if predictions["y_true"].nunique() < 2:
            continue
        frac_pos, mean_pred = calibration_curve(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["probability"].to_numpy(dtype=float),
            n_bins=8,
            strategy="quantile",
        )
        axis.plot(mean_pred, frac_pos, marker="o", label=str(result["model"]))
    axis.set_title(f"{dataset.upper()} calibration")
    axis.set_xlabel("mean predicted probability")
    axis.set_ylabel("fraction positive")
    axis.legend()
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures_dir / "calibration_curve.png")
    plt.close(figure)


def _plot_metropt_timeline(
    modeling: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    events: tuple[FailureEvent, ...],
    figures_dir: Path,
    best_model: str,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    best_predictions = predictions[predictions["model"].eq(best_model)].copy()
    if best_predictions.empty:
        return
    sampled_modeling = _sample_evenly(modeling, 3000)
    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes[0].plot(sampled_modeling["timestamp"], sampled_modeling["TP2__current"], label="TP2")
    axes[0].plot(
        sampled_modeling["timestamp"],
        sampled_modeling["Oil_temperature__current"],
        label="Oil temperature",
        alpha=0.8,
    )
    axes[0].set_title("MetroPT sensor timeline")
    axes[0].legend(loc="upper right")

    axes[1].plot(
        sampled_modeling["timestamp"],
        sampled_modeling["Motor_current__current"],
        color="#805ad5",
        label="Motor current",
    )
    axes[1].set_title("Motor current")
    axes[1].legend(loc="upper right")

    axes[2].plot(
        best_predictions["timestamp"],
        best_predictions["probability"],
        color="#2454a6",
        linewidth=0.8,
        label=f"{best_model} probability",
    )
    threshold = float(best_predictions["threshold"].iloc[0])
    axes[2].axhline(threshold, color="#a83232", linestyle="--", label="warning threshold")
    axes[2].set_title("Predicted failure probability")
    axes[2].set_ylabel("probability")
    axes[2].legend(loc="upper right")
    for axis in axes:
        _shade_failure_events(axis, events)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # type: ignore[no-untyped-call]
    figure.savefig(figures_dir / "metropt_sensor_probability_timeline.png")
    plt.close(figure)


def _plot_lead_time(episodes: pd.DataFrame, figures_dir: Path) -> None:
    if episodes.empty:
        return
    frame = episodes.copy()
    frame["lead_time_hours"] = pd.to_numeric(frame["lead_time_hours"], errors="coerce")
    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for model, group in frame.groupby("model", observed=True):
        axis.bar(
            [f"{model}\n{event}" for event in group["event_id"]],
            group["lead_time_hours"].fillna(0.0),
            label=str(model),
        )
    axis.set_title("Lead time by failure episode")
    axis.set_ylabel("hours before failure")
    axis.tick_params(axis="x", labelrotation=35)
    figure.savefig(figures_dir / "lead_time_by_episode.png")
    plt.close(figure)


def _shade_failure_events(axis: Axes, events: tuple[FailureEvent, ...]) -> None:
    for event in events:
        axis.axvspan(event.start, event.end, color="#ef4444", alpha=0.12, linewidth=0)


def _active_failure_mask(
    timestamps: pd.Series,
    failure_events: tuple[FailureEvent, ...],
) -> np.ndarray:
    active = np.zeros(timestamps.shape[0], dtype=bool)
    for event in failure_events:
        active |= ((timestamps >= event.start) & (timestamps <= event.end)).to_numpy()
    return active


def _events_in_split(
    events: tuple[FailureEvent, ...],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[FailureEvent, ...]:
    return tuple(event for event in events if start <= event.start <= end)


def _causal_window_rate(
    *,
    values: pd.Series,
    timestamps: pd.Series,
    window: pd.Timedelta,
    min_periods: int,
) -> np.ndarray:
    """Estimate trailing-window change per second without future observations."""

    numeric = values.to_numpy(dtype=float)
    time_ns = pd.to_datetime(timestamps, errors="raise").astype("int64").to_numpy()
    window_ns = int(window.value)
    start_indices = np.searchsorted(time_ns, time_ns - window_ns, side="left")
    positions = np.arange(numeric.shape[0], dtype=int)
    counts = positions - start_indices + 1
    elapsed_seconds = (time_ns - time_ns[start_indices]) / 1_000_000_000
    valid = (
        (counts >= min_periods)
        & (elapsed_seconds > 0)
        & np.isfinite(numeric)
        & np.isfinite(numeric[start_indices])
    )
    result = np.full(numeric.shape[0], np.nan, dtype=np.float32)
    result[valid] = (
        (numeric[valid] - numeric[start_indices[valid]]) / elapsed_seconds[valid]
    ).astype(np.float32)
    return result


def _window_suffix(value: str) -> str:
    return value.replace(" ", "").replace("-", "_")


def _sample_evenly(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0:
        return frame.iloc[0:0].copy()
    if frame.shape[0] <= max_rows:
        return frame.copy()
    indices = np.linspace(0, frame.shape[0] - 1, max_rows, dtype=int)
    return frame.iloc[indices].copy()


def _positive_weight(y: np.ndarray) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        return 1.0
    return float(max(negatives / positives, 1.0))


def _duration_days(timestamps: pd.Series) -> float:
    if timestamps.empty:
        return 0.0
    return float((timestamps.max() - timestamps.min()).total_seconds() / 86400)


def _best_metrics_record(metrics: pd.DataFrame) -> dict[str, Any]:
    ordered = metrics.sort_values(
        ["f1", "pr_auc", "recall", "false_positive_rate"],
        ascending=[False, False, False, True],
    )
    return cast(dict[str, Any], ordered.iloc[0].to_dict())


def _records_by_key(frame: pd.DataFrame, *, key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        records[str(record[key])] = cast(dict[str, Any], record)
    return records


def _write_dataframe(frame: pd.DataFrame, path_without_suffix: Path) -> Path:
    parquet_path = path_without_suffix.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = path_without_suffix.with_suffix(".csv.gz")
        frame.to_csv(csv_path, index=False)
        return csv_path


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping")
    return dict(value)


def _load_ai4i_config(payload: dict[str, Any]) -> AI4IExperimentConfig:
    return AI4IExperimentConfig(
        train_fraction=float(payload.get("train_fraction", 0.7)),
        validation_fraction=float(payload.get("validation_fraction", 0.15)),
        threshold_grid=_tuple_floats(payload.get("threshold_grid"), DEFAULT_THRESHOLD_GRID),
    )


def _load_metropt_config(payload: dict[str, Any]) -> MetroPTExperimentConfig:
    default = MetroPTExperimentConfig()
    events_payload = payload.get("failure_events")
    events = (
        tuple(FailureEvent(**event) for event in events_payload)
        if events_payload is not None
        else default.failure_events
    )
    return MetroPTExperimentConfig(
        horizon_hours=float(payload.get("horizon_hours", default.horizon_hours)),
        prediction_stride_rows=int(
            payload.get("prediction_stride_rows", default.prediction_stride_rows)
        ),
        train_end=str(payload.get("train_end", default.train_end)),
        validation_end=str(payload.get("validation_end", default.validation_end)),
        lag_steps=_tuple_ints(payload.get("lag_steps"), default.lag_steps),
        rolling_windows=_tuple_strings(payload.get("rolling_windows"), default.rolling_windows),
        trend_window=str(payload.get("trend_window", default.trend_window)),
        min_periods=int(payload.get("min_periods", default.min_periods)),
        max_train_rows=int(payload.get("max_train_rows", default.max_train_rows)),
        negative_to_positive_ratio=int(
            payload.get("negative_to_positive_ratio", default.negative_to_positive_ratio)
        ),
        threshold_grid=_tuple_floats(payload.get("threshold_grid"), default.threshold_grid),
        failure_events=events,
    )


def _tuple_floats(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of floats")
    return tuple(float(item) for item in value)


def _tuple_ints(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of integers")
    return tuple(int(item) for item in value)


def _tuple_strings(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of strings")
    return tuple(str(item) for item in value)


def _validate_split_fractions(train_fraction: float, validation_fraction: float) -> None:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")


def _validate_threshold_grid(thresholds: tuple[float, ...]) -> None:
    if not thresholds:
        raise ValueError("threshold_grid must not be empty")
    if any(threshold <= 0 or threshold >= 1 for threshold in thresholds):
        raise ValueError("threshold_grid values must be in (0, 1)")


def _write_phase_report(
    run_dir: Path,
    *,
    ai4i_result: DatasetResult,
    metropt_result: DatasetResult,
) -> None:
    ai_best = ai4i_result.metrics["best_model"]
    metro_best = metropt_result.metrics["best_model"]
    lines = [
        "# Failure Prediction Phase 7",
        "",
        "## AI4I",
        "",
        f"- Target: {ai4i_result.metrics['target_definition']}",
        f"- Split: {ai4i_result.metrics['split_policy']}",
        f"- Best model: {ai_best['model']}",
        f"- F1: {_fmt(ai_best.get('f1'))}",
        f"- ROC-AUC: {_fmt(ai_best.get('roc_auc'))}",
        f"- PR-AUC: {_fmt(ai_best.get('pr_auc'))}",
        "",
        "## MetroPT",
        "",
        f"- Target: {metropt_result.metrics['target_definition']}",
        f"- Split: {metropt_result.metrics['split_policy']}",
        f"- Best model: {metro_best['model']}",
        f"- F1: {_fmt(metro_best.get('f1'))}",
        f"- ROC-AUC: {_fmt(metro_best.get('roc_auc'))}",
        f"- PR-AUC: {_fmt(metro_best.get('pr_auc'))}",
        f"- Warning coverage: {_fmt(metro_best.get('warning_coverage'))}",
        f"- Median lead time hours: {_fmt(metro_best.get('median_lead_time_hours'))}",
        f"- False alarm rate: {_fmt(metro_best.get('false_alarm_rate'))}",
        "",
        "SHAP artifacts describe model-contributing features, not causal root causes.",
    ]
    (run_dir / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
