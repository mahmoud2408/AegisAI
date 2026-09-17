"""Phase 5 NAB robustness and SMD planning analyses."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from aegis_ai.data.dataset_registry import project_root
from aegis_ai.evaluation.anomaly import (
    compute_binary_metrics,
    compute_detection_delay,
    compute_grouped_detection_delay,
    positive_segments,
)
from aegis_ai.features.timeseries import FEATURE_COLUMNS, feature_matrix
from aegis_ai.ml.anomaly.autoencoder import DenseAutoencoderDetector
from aegis_ai.ml.anomaly.isolation_forest import IsolationForestDetector
from aegis_ai.ml.anomaly.nab_deep_experiment import (
    MODEL_NAMES,
    ModelName,
    NabDeepExperimentConfig,
    _evaluate_dense_autoencoder,
    _evaluate_lstm_autoencoder,
    _prepare_series,
    _safe_filename,
    _series_metric_row,
    load_nab_deep_config,
)
from aegis_ai.ml.anomaly.nab_experiment import NabSeries, labels_for_series, load_processed_nab
from aegis_ai.ml.anomaly.smd_profile import profile_smd_dataset, write_smd_profile_artifacts

ThresholdStrategyName = Literal["train_p99", "validation_p99", "train_mad"]
ComplementarityLabel = Literal["both", "isolation_only", "dense_only", "neither"]


@dataclass(frozen=True)
class NabRobustnessConfig:
    """Configuration for the Phase 5 diagnostic analysis."""

    threshold_percentile: float = 99.0
    robust_mad_multiplier: float = 6.0
    window_sizes: tuple[int, ...] = (16, 32, 64)
    window_sensitivity_series: int = 6
    window_sensitivity_epochs: int = 2
    context_points: int = 160

    def __post_init__(self) -> None:
        if not 0 < self.threshold_percentile < 100:
            raise ValueError("threshold_percentile must be in (0, 100)")
        if self.robust_mad_multiplier <= 0:
            raise ValueError("robust_mad_multiplier must be positive")
        if not self.window_sizes or any(size < 2 for size in self.window_sizes):
            raise ValueError("window_sizes must contain values >= 2")
        if self.window_sensitivity_series < 1:
            raise ValueError("window_sensitivity_series must be positive")
        if self.window_sensitivity_epochs < 1:
            raise ValueError("window_sensitivity_epochs must be positive")
        if self.context_points < 10:
            raise ValueError("context_points must be at least 10")


@dataclass(frozen=True)
class NabRobustnessResult:
    """File locations for a completed Phase 5 analysis."""

    run_dir: Path
    smd_profile_dir: Path
    summary: dict[str, Any]


def run_nab_robustness_analysis(
    *,
    root: Path | None = None,
    phase4_run_dir: Path | None = None,
    output_root: Path | None = None,
    smd_output_root: Path | None = None,
    run_id: str | None = None,
    smd_run_id: str | None = None,
    config_path: Path | None = None,
    config: NabRobustnessConfig | None = None,
) -> NabRobustnessResult:
    """Run Phase 5 analysis from existing Phase 4 artifacts."""

    root = root or project_root()
    cfg = config or NabRobustnessConfig()
    base_config_path = config_path or root / "configs" / "experiments" / "nab_deep_anomaly.yaml"
    phase4_run_dir = phase4_run_dir or (
        root / "experiments" / "anomaly" / "nab" / "deep_autoencoders" / "run_phase4_deep_anomaly"
    )
    _require_phase4_artifacts(phase4_run_dir)

    run_dir = _create_run_dir(
        output_root or root / "experiments" / "anomaly" / "nab" / "robustness",
        run_id,
    )
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    smd_profile_dir = _create_run_dir(
        smd_output_root or root / "experiments" / "smd" / "profile",
        smd_run_id,
    )

    predictions = _read_predictions(phase4_run_dir / "predictions.csv")
    per_series = _read_per_series_metrics(phase4_run_dir / "per_series_metrics.csv")
    metrics = _read_json(phase4_run_dir / "metrics.json")
    error_analysis = _read_json(phase4_run_dir / "error_analysis.json")
    ablation = _read_json(phase4_run_dir / "ablation_results.json")
    phase4_config = load_nab_deep_config(base_config_path)
    series_items = load_processed_nab(root)

    series_analysis = build_series_error_table(series_items, per_series, predictions)
    robustness_summary = build_robustness_summary(per_series, series_analysis)
    official_scoring = review_official_nab_scoring(root)
    threshold_result = analyze_threshold_sensitivity(
        root=root,
        phase4_run_dir=phase4_run_dir,
        series_items=series_items,
        base_config=phase4_config,
        config=cfg,
    )
    complementarity = analyze_isolation_dense_complementarity(
        predictions,
        threshold_result["score_arrays"],
        config=cfg,
    )
    failure_examples = build_failure_case_analysis(
        predictions=predictions,
        error_analysis=error_analysis,
        output_dir=run_dir,
        config=cfg,
    )
    window_sensitivity = analyze_window_size_sensitivity(
        series_items=series_items,
        base_config=phase4_config,
        output_dir=run_dir,
        config=cfg,
    )
    smd_profile = profile_smd_dataset(root)
    write_smd_profile_artifacts(smd_profile, smd_profile_dir)

    threshold_per_series = threshold_result["per_series"]
    threshold_global = threshold_result["global"]
    pr_tradeoff = threshold_result["precision_recall_tradeoff"]
    score_cache = threshold_result["score_cache"]
    ensemble = complementarity["ensemble"]
    complementarity_summary = complementarity["summary"]
    complementarity_windows = complementarity["window_records"]
    complementarity_series = complementarity["series_records"]
    window_global = window_sensitivity["global"]
    window_per_series = window_sensitivity["per_series"]

    series_analysis.to_csv(run_dir / "per_series_error_analysis.csv", index=False)
    threshold_per_series.to_csv(run_dir / "threshold_sensitivity_per_series.csv", index=False)
    threshold_global.to_csv(run_dir / "threshold_sensitivity_global.csv", index=False)
    pr_tradeoff.to_csv(run_dir / "precision_recall_tradeoff.csv", index=False)
    complementarity_summary.to_csv(run_dir / "if_dense_complementarity_summary.csv", index=False)
    complementarity_series.to_csv(run_dir / "if_dense_complementarity_by_series.csv", index=False)
    complementarity_windows.to_csv(run_dir / "if_dense_window_overlap.csv", index=False)
    ensemble.to_csv(run_dir / "if_dense_ensemble_diagnostic.csv", index=False)
    window_global.to_csv(run_dir / "window_sensitivity_global.csv", index=False)
    window_per_series.to_csv(run_dir / "window_sensitivity_per_series.csv", index=False)
    score_cache.to_csv(run_dir / "phase4_score_cache_manifest.csv", index=False)

    _plot_series_classification(run_dir, series_analysis)
    _plot_threshold_sensitivity(run_dir, threshold_global)
    _plot_precision_recall_tradeoff(run_dir, pr_tradeoff)
    _plot_complementarity(run_dir, complementarity_summary)
    _plot_window_sensitivity(run_dir, window_global)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_phase4_run_dir": str(phase4_run_dir),
        "config": asdict(cfg),
        "phase4_global_metrics": metrics.get("global", {}),
        "phase4_ablation_scope": ablation.get("scope"),
        "series_classification_counts": _value_counts(series_analysis["difficulty_class"]),
        "robustness": robustness_summary,
        "official_nab_scoring": official_scoring,
        "threshold_best_by_f1": _best_threshold_rows(threshold_global),
        "complementarity": _frame_records(complementarity_summary),
        "ensemble_diagnostic": _frame_records(ensemble),
        "window_sensitivity_best_by_f1": _best_window_rows(window_global),
        "failure_examples": failure_examples,
        "smd_profile_dir": str(smd_profile_dir),
        "smd_profile": {
            key: value for key, value in smd_profile.items() if key not in {"machine_profile"}
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_run_report(run_dir, smd_profile_dir, summary)

    return NabRobustnessResult(run_dir=run_dir, smd_profile_dir=smd_profile_dir, summary=summary)


def classify_series_difficulty(
    *,
    test_positive_points: int,
    best_f1: float,
    best_recall: float,
    max_false_positive_rate: float,
) -> str:
    """Classify one series using explicit Phase 5 diagnostic criteria."""

    if test_positive_points > 0:
        if best_f1 >= 0.5 and best_recall >= 0.4:
            return "easy"
        if best_f1 >= 0.2:
            return "moderate"
        if best_f1 > 0:
            return "difficult"
        return "complete_failure"
    if max_false_positive_rate <= 0.01:
        return "easy"
    if max_false_positive_rate <= 0.05:
        return "moderate"
    if max_false_positive_rate <= 0.15:
        return "difficult"
    return "complete_failure"


def threshold_from_scores(
    strategy: ThresholdStrategyName,
    *,
    train_scores: np.ndarray,
    validation_scores: np.ndarray,
    percentile: float,
    mad_multiplier: float,
) -> float:
    """Select an unsupervised threshold from train or validation scores."""

    train = _finite_scores(train_scores)
    validation = _finite_scores(validation_scores)
    if strategy == "train_p99":
        return _percentile_threshold(train, percentile)
    if strategy == "validation_p99":
        source = validation if validation.size else train
        return _percentile_threshold(source, percentile)
    if strategy == "train_mad":
        median = float(np.median(train))
        mad = float(np.median(np.abs(train - median)))
        robust_sigma = 1.4826 * mad
        threshold = median + mad_multiplier * robust_sigma
        if not np.isfinite(threshold) or threshold <= median:
            return _percentile_threshold(train, percentile)
        return float(threshold)
    raise ValueError(f"unsupported threshold strategy: {strategy}")


def empirical_percentile_scores(reference_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map scores to empirical percentile ranks relative to a training reference."""

    reference = np.sort(_finite_scores(reference_scores))
    value_array = np.asarray(values, dtype=float)
    if reference.size == 0:
        raise ValueError("reference_scores must contain finite values")
    ranks = np.searchsorted(reference, value_array, side="right")
    return ranks.astype(float) / float(reference.size)


def build_series_error_table(
    series_items: Iterable[NabSeries],
    per_series: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per NAB series with point/window statistics and model metrics."""

    rows: list[dict[str, Any]] = []
    metrics_by_entity_model = {
        (str(row["entity_id"]), str(row["model"])): row for _, row in per_series.iterrows()
    }
    predictions_by_entity = {
        str(entity_id): group.sort_values("timestamp").reset_index(drop=True)
        for entity_id, group in predictions.groupby("entity_id", sort=True)
    }
    for series in sorted(series_items, key=lambda item: item.entity_id):
        labels = labels_for_series(series)
        prediction_group = predictions_by_entity.get(series.entity_id)
        if prediction_group is None:
            test_positive_points = 0
            test_anomaly_windows = 0
            test_rows = 0
        else:
            test_labels = prediction_group["y_true"].to_numpy(dtype=int)
            test_positive_points = int(test_labels.sum())
            test_anomaly_windows = len(positive_segments(test_labels))
            test_rows = int(len(prediction_group))

        row: dict[str, Any] = {
            "entity_id": series.entity_id,
            "metric_name": series.metric_name,
            "observations": int(len(series.frame)),
            "anomaly_windows": int(len(series.windows)),
            "anomaly_points": int(labels.sum()),
            "test_rows": test_rows,
            "test_positive_points": test_positive_points,
            "test_anomaly_windows": test_anomaly_windows,
        }
        best_f1 = 0.0
        best_recall = 0.0
        best_model = "none"
        max_false_positive_rate = 0.0
        for model in MODEL_NAMES:
            metric_row = metrics_by_entity_model.get((series.entity_id, model))
            if metric_row is None:
                _add_missing_model_metrics(row, model)
                continue
            f1 = _float_value(metric_row.get("f1"))
            recall = _float_value(metric_row.get("recall"))
            false_positive_rate = _float_value(metric_row.get("false_positive_rate"))
            row[f"{model}_precision"] = _float_value(metric_row.get("precision"))
            row[f"{model}_recall"] = recall
            row[f"{model}_f1"] = f1
            row[f"{model}_pr_auc"] = _maybe_float(metric_row.get("pr_auc"))
            row[f"{model}_false_positive_rate"] = false_positive_rate
            row[f"{model}_detected_windows"] = int(
                _float_value(metric_row.get("detection_detected_windows"))
            )
            row[f"{model}_missed_windows"] = int(
                _float_value(metric_row.get("detection_missed_windows"))
            )
            row[f"{model}_median_delay_steps"] = _maybe_float(
                metric_row.get("detection_median_delay_steps")
            )
            if f1 > best_f1 or (f1 == best_f1 and recall > best_recall):
                best_f1 = f1
                best_recall = recall
                best_model = model
            max_false_positive_rate = max(max_false_positive_rate, false_positive_rate)

        row["best_model"] = best_model
        row["best_f1"] = best_f1
        row["best_recall"] = best_recall
        row["max_false_positive_rate"] = max_false_positive_rate
        row["difficulty_class"] = classify_series_difficulty(
            test_positive_points=test_positive_points,
            best_f1=best_f1,
            best_recall=best_recall,
            max_false_positive_rate=max_false_positive_rate,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_robustness_summary(
    per_series: pd.DataFrame,
    series_analysis: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize robustness across all NAB series and positive-test series."""

    summary: dict[str, Any] = {
        "series_count": int(series_analysis["entity_id"].nunique()),
        "positive_test_series": int((series_analysis["test_positive_points"] > 0).sum()),
        "difficulty_counts": _value_counts(series_analysis["difficulty_class"]),
        "models": {},
    }
    for model in MODEL_NAMES:
        frame = per_series[per_series["model"].eq(model)].copy()
        positives = frame[frame["positives"] > 0]
        f1 = pd.to_numeric(frame["f1"], errors="coerce")
        detected = pd.to_numeric(frame["detection_detected_windows"], errors="coerce")
        model_summary = {
            "series_evaluated": int(len(frame)),
            "positive_test_series": int(len(positives)),
            "nonzero_f1_series": int((f1 > 0).sum()),
            "nonzero_f1_pct_all": _safe_divide(float((f1 > 0).sum()), float(len(frame))),
            "nonzero_f1_pct_positive": _safe_divide(
                float((pd.to_numeric(positives["f1"], errors="coerce") > 0).sum()),
                float(len(positives)),
            ),
            "successful_detection_series": int((detected > 0).sum()),
            "mean_f1": _maybe_float(f1.mean()),
            "median_f1": _maybe_float(f1.median()),
            "variance_f1": _maybe_float(f1.var(ddof=0)),
            "std_f1": _maybe_float(f1.std(ddof=0)),
            "worst_f1": _maybe_float(f1.min()),
            "best_f1": _maybe_float(f1.max()),
            "positive_zero_f1_series": int(
                (pd.to_numeric(positives["f1"], errors="coerce") == 0).sum()
            ),
        }
        summary["models"][model] = model_summary
    return summary


def review_official_nab_scoring(root: Path) -> dict[str, Any]:
    """Report whether exact official NAB scoring can be computed from local artifacts."""

    raw_root = root / "data" / "raw" / "nab"
    observed = sorted(path.relative_to(raw_root).as_posix() for path in raw_root.iterdir())
    required_candidates = [
        raw_root / "nab",
        raw_root / "config",
        raw_root / "scripts",
        raw_root / "run.py",
    ]
    has_candidate_scorer = any(path.exists() for path in required_candidates)
    return {
        "implemented": False,
        "can_compute_exact_official_score": has_candidate_scorer,
        "observed_top_level_entries": observed,
        "phase5_decision": (
            "Exact official NAB scoring is not reported. The local mirror contains NAB data and "
            "labels but not the official scorer/application-profile implementation needed to "
            "reproduce normalized NAB scores exactly."
        ),
        "reported_metrics": (
            "Phase 5 reports generic point metrics, detection-delay metrics, and window hit "
            "counts only."
        ),
    }


def analyze_threshold_sensitivity(
    *,
    root: Path,
    phase4_run_dir: Path,
    series_items: list[NabSeries],
    base_config: NabDeepExperimentConfig,
    config: NabRobustnessConfig,
) -> dict[str, Any]:
    """Evaluate threshold sensitivity for saved IF and Dense AE models."""

    strategies: tuple[ThresholdStrategyName, ...] = ("train_p99", "validation_p99", "train_mad")
    per_series_rows: list[dict[str, Any]] = []
    score_manifest_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    tradeoff_frames: list[pd.DataFrame] = []
    cache: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "isolation_forest": {},
        "dense_autoencoder": {},
    }
    del root

    for series in sorted(series_items, key=lambda item: item.entity_id):
        prepared = _prepare_series(series, base_config)
        safe_name = _safe_filename(series.entity_id)
        if_scores = _score_isolation_forest(
            phase4_run_dir / "models" / "isolation_forest" / f"{safe_name}.joblib",
            prepared,
        )
        dense_scores = _score_dense_autoencoder(
            phase4_run_dir / "models" / "dense_autoencoder" / f"{safe_name}.pt",
            prepared,
        )
        cache["isolation_forest"][series.entity_id] = if_scores
        cache["dense_autoencoder"][series.entity_id] = dense_scores
        y_true = prepared.test_eval["y_true"].to_numpy(dtype=int)
        timestamps = prepared.test_eval["timestamp"].tolist()
        for model_name, score_sets in (
            ("isolation_forest", if_scores),
            ("dense_autoencoder", dense_scores),
        ):
            score_manifest_rows.append(
                {
                    "entity_id": series.entity_id,
                    "model": model_name,
                    "train_scores": int(score_sets["train"].shape[0]),
                    "validation_scores": int(score_sets["validation"].shape[0]),
                    "test_scores": int(score_sets["test"].shape[0]),
                }
            )
            for strategy in strategies:
                threshold = threshold_from_scores(
                    strategy,
                    train_scores=score_sets["train"],
                    validation_scores=score_sets["validation"],
                    percentile=config.threshold_percentile,
                    mad_multiplier=config.robust_mad_multiplier,
                )
                y_pred = (score_sets["test"] >= threshold).astype(int)
                metrics = compute_binary_metrics(y_true, y_pred, score_sets["test"]).to_dict()
                delay = compute_detection_delay(y_true, y_pred, timestamps).to_dict()
                per_series_rows.append(
                    {
                        "entity_id": series.entity_id,
                        "metric_name": series.metric_name,
                        "model": model_name,
                        "threshold_strategy": strategy,
                        "threshold": threshold,
                        **metrics,
                        **{f"detection_{key}": value for key, value in delay.items()},
                    }
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "entity_id": series.entity_id,
                            "timestamp": timestamps,
                            "y_true": y_true,
                            "model": model_name,
                            "threshold_strategy": strategy,
                            "score": score_sets["test"],
                            "prediction": y_pred,
                        }
                    )
                )

    threshold_per_series = pd.DataFrame(per_series_rows)
    threshold_global = _aggregate_threshold_metrics(pd.concat(prediction_frames, ignore_index=True))
    score_manifest = pd.DataFrame(score_manifest_rows)
    for model_name in ("isolation_forest", "dense_autoencoder"):
        y_true_all = []
        scores_all = []
        for series_id in sorted(cache[model_name]):
            prepared_scores = cache[model_name][series_id]
            y_true_all.append(_truth_for_entity(series_id, series_items, base_config))
            scores_all.append(prepared_scores["test"])
        tradeoff_frames.append(
            _precision_recall_tradeoff_frame(
                np.concatenate(y_true_all),
                np.concatenate(scores_all),
                model_name=model_name,
            )
        )

    return {
        "per_series": threshold_per_series,
        "global": threshold_global,
        "precision_recall_tradeoff": pd.concat(tradeoff_frames, ignore_index=True),
        "score_cache": score_manifest,
        "score_arrays": cache,
    }


def analyze_isolation_dense_complementarity(
    predictions: pd.DataFrame,
    score_arrays: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    config: NabRobustnessConfig,
) -> dict[str, pd.DataFrame]:
    """Measure whether IF and Dense AE fire on different points and windows."""

    del config
    point_rows = []
    series_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for entity_id, group in predictions.groupby("entity_id", sort=True):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        truth = ordered["y_true"].to_numpy(dtype=int)
        if_pred = ordered["isolation_forest_prediction"].to_numpy(dtype=int)
        dense_pred = ordered["dense_autoencoder_prediction"].to_numpy(dtype=int)
        both = (if_pred == 1) & (dense_pred == 1)
        if_only = (if_pred == 1) & (dense_pred == 0)
        dense_only = (if_pred == 0) & (dense_pred == 1)
        neither = (if_pred == 0) & (dense_pred == 0)
        union = (if_pred == 1) | (dense_pred == 1)
        series_rows.append(
            {
                "entity_id": entity_id,
                "points": int(len(ordered)),
                "positive_points": int(truth.sum()),
                "if_detections": int(if_pred.sum()),
                "dense_detections": int(dense_pred.sum()),
                "both_detections": int(both.sum()),
                "if_only_detections": int(if_only.sum()),
                "dense_only_detections": int(dense_only.sum()),
                "jaccard": _safe_divide(float(both.sum()), float(union.sum())),
                "if_only_true_positives": int((if_only & (truth == 1)).sum()),
                "dense_only_true_positives": int((dense_only & (truth == 1)).sum()),
                "if_only_false_positives": int((if_only & (truth == 0)).sum()),
                "dense_only_false_positives": int((dense_only & (truth == 0)).sum()),
            }
        )
        for window_index, (start, end) in enumerate(positive_segments(truth)):
            if_hit = bool(np.any(if_pred[start:end] == 1))
            dense_hit = bool(np.any(dense_pred[start:end] == 1))
            label = _window_overlap_label(if_hit=if_hit, dense_hit=dense_hit)
            window_rows.append(
                {
                    "entity_id": entity_id,
                    "window_index": window_index,
                    "window_start": str(ordered["timestamp"].iloc[start]),
                    "window_end": str(ordered["timestamp"].iloc[end - 1]),
                    "points": int(end - start),
                    "if_detected": if_hit,
                    "dense_detected": dense_hit,
                    "overlap_label": label,
                }
            )
        point_rows.append(
            {
                "entity_id": entity_id,
                "both": int(both.sum()),
                "isolation_only": int(if_only.sum()),
                "dense_only": int(dense_only.sum()),
                "neither": int(neither.sum()),
                "union": int(union.sum()),
            }
        )

    summary = _complementarity_summary(pd.DataFrame(point_rows), pd.DataFrame(window_rows))
    ensemble = _ensemble_from_score_arrays(predictions, score_arrays)
    return {
        "summary": summary,
        "series_records": pd.DataFrame(series_rows),
        "window_records": pd.DataFrame(window_rows),
        "ensemble": ensemble,
    }


def build_failure_case_analysis(
    *,
    predictions: pd.DataFrame,
    error_analysis: dict[str, Any],
    output_dir: Path,
    config: NabRobustnessConfig,
) -> dict[str, Any]:
    """Build representative failure cases with observable context statistics."""

    cases: list[dict[str, Any]] = []
    chosen_specs = [
        ("false_positive", "lstm_autoencoder", "false_positive_examples", 0),
        ("missed_window", "dense_autoencoder", "missed_window_examples", 0),
        ("missed_window", "isolation_forest", "missed_window_examples", 0),
        ("delayed_window", "lstm_autoencoder", "delayed_window_examples", 0),
        ("delayed_window", "dense_autoencoder", "delayed_window_examples", 0),
    ]
    for case_type, model_name, key, index in chosen_specs:
        items = cast(list[dict[str, Any]], error_analysis.get(model_name, {}).get(key, []))
        if index >= len(items):
            continue
        case = _case_context(
            predictions=predictions,
            model_name=model_name,
            case_type=case_type,
            item=items[index],
            context_points=config.context_points,
        )
        cases.append(case)
        _plot_failure_case(output_dir, predictions, case, model_name=model_name)

    noisy_normal = _noisy_normal_case(predictions, context_points=config.context_points)
    if noisy_normal:
        cases.append(noisy_normal)
        _plot_failure_case(
            output_dir,
            predictions,
            noisy_normal,
            model_name=str(noisy_normal["model"]),
        )
    return {"cases": cases}


def analyze_window_size_sensitivity(
    *,
    series_items: list[NabSeries],
    base_config: NabDeepExperimentConfig,
    output_dir: Path,
    config: NabRobustnessConfig,
) -> dict[str, pd.DataFrame]:
    """Run a bounded neural model window-size diagnostic on labeled NAB series."""

    selected = _select_labeled_series_for_window_sensitivity(
        series_items,
        base_config,
        max_series=config.window_sensitivity_series,
    )
    global_rows: list[dict[str, Any]] = []
    per_series_rows: list[dict[str, Any]] = []
    for window_size in config.window_sizes:
        variant_config = _window_variant_config(base_config, window_size, config)
        prediction_frames: dict[str, list[pd.DataFrame]] = {
            "dense_autoencoder": [],
            "lstm_autoencoder": [],
        }
        for series in selected:
            prepared = _prepare_series(series, variant_config)
            dense = _evaluate_dense_autoencoder(
                prepared,
                variant_config,
                output_dir,
                save_model=False,
            )
            lstm = _evaluate_lstm_autoencoder(
                prepared,
                variant_config,
                output_dir,
                save_model=False,
            )
            for outcome in (dense, lstm):
                row = _series_metric_row(prepared, outcome)
                row["window_size"] = window_size
                row["diagnostic_epochs"] = config.window_sensitivity_epochs
                per_series_rows.append(row)
                prediction_frames[outcome.model_name].append(
                    pd.DataFrame(
                        {
                            "entity_id": prepared.series.entity_id,
                            "timestamp": prepared.test_eval["timestamp"].tolist(),
                            "y_true": prepared.test_eval["y_true"].to_numpy(dtype=int),
                            "prediction": outcome.predictions,
                            "score": outcome.scores,
                        }
                    )
                )
        for model_name, frames in prediction_frames.items():
            combined = pd.concat(frames, ignore_index=True)
            y_true = combined["y_true"].to_numpy(dtype=int)
            y_pred = combined["prediction"].to_numpy(dtype=int)
            scores = combined["score"].to_numpy(dtype=float)
            metrics = compute_binary_metrics(y_true, y_pred, scores).to_dict()
            delay = compute_grouped_detection_delay(
                combined,
                y_true_column="y_true",
                y_pred_column="prediction",
            ).to_dict()
            global_rows.append(
                {
                    "model": model_name,
                    "window_size": window_size,
                    "series_count": len(selected),
                    "diagnostic_epochs": config.window_sensitivity_epochs,
                    **metrics,
                    **{f"detection_{key}": value for key, value in delay.items()},
                }
            )
    return {"global": pd.DataFrame(global_rows), "per_series": pd.DataFrame(per_series_rows)}


def _require_phase4_artifacts(phase4_run_dir: Path) -> None:
    required = [
        phase4_run_dir / "predictions.csv",
        phase4_run_dir / "per_series_metrics.csv",
        phase4_run_dir / "metrics.json",
        phase4_run_dir / "error_analysis.json",
        phase4_run_dir / "models" / "isolation_forest",
        phase4_run_dir / "models" / "dense_autoencoder",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Phase 4 artifacts: {[str(path) for path in missing]}")


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    numeric_columns = [
        "metric_value",
        "y_true",
        *[f"{model}_score" for model in MODEL_NAMES],
        *[f"{model}_prediction" for model in MODEL_NAMES],
        *[f"{model}_threshold" for model in MODEL_NAMES],
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _read_per_series_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in frame.columns:
        if column not in {"entity_id", "metric_name", "model"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _score_isolation_forest(path: Path, prepared: Any) -> dict[str, np.ndarray]:
    detector = IsolationForestDetector.load(path)
    return {
        "train": detector.score_samples(
            feature_matrix(prepared.train_threshold_frame, FEATURE_COLUMNS)
        ),
        "validation": detector.score_samples(feature_matrix(prepared.validation, FEATURE_COLUMNS)),
        "test": detector.score_samples(feature_matrix(prepared.test_eval, FEATURE_COLUMNS)),
    }


def _score_dense_autoencoder(path: Path, prepared: Any) -> dict[str, np.ndarray]:
    detector = DenseAutoencoderDetector.load(path)
    return {
        "train": detector.reconstruction_error(prepared.train_fit_windows),
        "validation": detector.reconstruction_error(prepared.validation_windows.windows),
        "test": detector.reconstruction_error(prepared.test_windows.windows),
    }


def _truth_for_entity(
    entity_id: str,
    series_items: list[NabSeries],
    base_config: NabDeepExperimentConfig,
) -> np.ndarray:
    series = next(item for item in series_items if item.entity_id == entity_id)
    prepared = _prepare_series(series, base_config)
    return cast(np.ndarray, prepared.test_eval["y_true"].to_numpy(dtype=int))


def _aggregate_threshold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, strategy), group in predictions.groupby(
        ["model", "threshold_strategy"],
        sort=True,
    ):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_pred = group["prediction"].to_numpy(dtype=int)
        scores = group["score"].to_numpy(dtype=float)
        metrics = compute_binary_metrics(y_true, y_pred, scores).to_dict()
        delay = compute_grouped_detection_delay(
            group,
            y_true_column="y_true",
            y_pred_column="prediction",
        ).to_dict()
        rows.append(
            {
                "model": model_name,
                "threshold_strategy": strategy,
                **metrics,
                **{f"detection_{key}": value for key, value in delay.items()},
            }
        )
    return pd.DataFrame(rows)


def _precision_recall_tradeoff_frame(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    model_name: str,
    max_rows: int = 300,
) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    threshold_values = np.append(thresholds, np.nan)
    frame = pd.DataFrame(
        {
            "model": model_name,
            "precision": precision,
            "recall": recall,
            "threshold": threshold_values,
            "diagnostic_uses_test_labels": True,
        }
    )
    if len(frame) <= max_rows:
        return frame
    indices = np.linspace(0, len(frame) - 1, max_rows).astype(int)
    return frame.iloc[indices].reset_index(drop=True)


def _complementarity_summary(points: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    point_totals = points[["both", "isolation_only", "dense_only", "neither", "union"]].sum()
    window_counts = _value_counts(windows["overlap_label"]) if not windows.empty else {}
    return pd.DataFrame(
        [
            {
                "scope": "point_predictions",
                "both": int(point_totals.get("both", 0)),
                "isolation_only": int(point_totals.get("isolation_only", 0)),
                "dense_only": int(point_totals.get("dense_only", 0)),
                "neither": int(point_totals.get("neither", 0)),
                "union": int(point_totals.get("union", 0)),
                "jaccard": _safe_divide(
                    float(point_totals.get("both", 0)),
                    float(point_totals.get("union", 0)),
                ),
                "window_both": window_counts.get("both", 0),
                "window_isolation_only": window_counts.get("isolation_only", 0),
                "window_dense_only": window_counts.get("dense_only", 0),
                "window_neither": window_counts.get("neither", 0),
            }
        ]
    )


def _ensemble_from_score_arrays(
    predictions: pd.DataFrame,
    score_arrays: dict[str, dict[str, dict[str, np.ndarray]]],
) -> pd.DataFrame:
    rows = []
    test_frames: list[pd.DataFrame] = []
    for entity_id, group in predictions.groupby("entity_id", sort=True):
        if_scores = score_arrays["isolation_forest"][str(entity_id)]
        dense_scores = score_arrays["dense_autoencoder"][str(entity_id)]
        test_score = np.maximum(
            empirical_percentile_scores(if_scores["train"], if_scores["test"]),
            empirical_percentile_scores(dense_scores["train"], dense_scores["test"]),
        )
        if len(group) != test_score.shape[0]:
            raise ValueError(f"ensemble score/test length mismatch for {entity_id}")
        test_frame = group.loc[:, ["entity_id", "timestamp", "y_true"]].copy()
        test_frame["if_dense_max_percentile_score"] = test_score
        test_frames.append(test_frame)

    threshold = 0.99
    frame = pd.concat(test_frames, ignore_index=True)
    frame["if_dense_max_percentile_prediction"] = (
        frame["if_dense_max_percentile_score"] >= threshold
    ).astype(int)
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["if_dense_max_percentile_prediction"].to_numpy(dtype=int)
    scores = frame["if_dense_max_percentile_score"].to_numpy(dtype=float)
    metrics = compute_binary_metrics(y_true, y_pred, scores).to_dict()
    delay = compute_grouped_detection_delay(
        frame,
        y_true_column="y_true",
        y_pred_column="if_dense_max_percentile_prediction",
    ).to_dict()
    rows.append(
        {
            "model": "if_dense_max_percentile",
            "threshold_source": (
                "0.99 train-calibrated empirical percentile for either IF or Dense score; "
                "diagnostic only and not registered as a production ensemble"
            ),
            "threshold": threshold,
            **metrics,
            **{f"detection_{key}": value for key, value in delay.items()},
        }
    )
    return pd.DataFrame(rows)


def _window_overlap_label(*, if_hit: bool, dense_hit: bool) -> ComplementarityLabel:
    if if_hit and dense_hit:
        return "both"
    if if_hit:
        return "isolation_only"
    if dense_hit:
        return "dense_only"
    return "neither"


def _case_context(
    *,
    predictions: pd.DataFrame,
    model_name: str,
    case_type: str,
    item: dict[str, Any],
    context_points: int,
) -> dict[str, Any]:
    entity_id = str(item["entity_id"])
    group = predictions[predictions["entity_id"].eq(entity_id)].sort_values("timestamp")
    group = group.reset_index(drop=True)
    timestamp_column = pd.to_datetime(group["timestamp"], errors="coerce")
    if "timestamp" in item:
        target = pd.to_datetime(item["timestamp"], errors="coerce")
        center = int((timestamp_column - target).abs().argmin())
        start = max(0, center - context_points // 2)
        end = min(len(group), center + context_points // 2)
        target_slice = group.iloc[[center]]
    else:
        window_start = pd.to_datetime(item["window_start"], errors="coerce")
        window_end = pd.to_datetime(item["window_end"], errors="coerce")
        mask = (timestamp_column >= window_start) & (timestamp_column <= window_end)
        indices = np.flatnonzero(mask.to_numpy())
        if indices.size == 0:
            center = len(group) // 2
            start = max(0, center - context_points // 2)
            end = min(len(group), center + context_points // 2)
            target_slice = group.iloc[start:end]
        else:
            start = max(0, int(indices[0]) - context_points // 3)
            end = min(len(group), int(indices[-1]) + context_points // 3)
            target_slice = group.iloc[int(indices[0]) : int(indices[-1]) + 1]
    context = group.iloc[start:end]
    score_column = f"{model_name}_score"
    threshold_column = f"{model_name}_threshold"
    prediction_column = f"{model_name}_prediction"
    threshold = float(group[threshold_column].dropna().iloc[0])
    return {
        "case_type": case_type,
        "model": model_name,
        "entity_id": entity_id,
        "start_timestamp": str(context["timestamp"].iloc[0]),
        "end_timestamp": str(context["timestamp"].iloc[-1]),
        "target_points": int(len(target_slice)),
        "context_points": int(len(context)),
        "anomaly_duration_points": int(target_slice["y_true"].sum()),
        "detections_in_target": int(target_slice[prediction_column].sum()),
        "metric_min": float(target_slice["metric_value"].min()),
        "metric_max": float(target_slice["metric_value"].max()),
        "metric_mean": float(target_slice["metric_value"].mean()),
        "context_metric_mean": float(context["metric_value"].mean()),
        "context_metric_std": float(context["metric_value"].std(ddof=0)),
        "local_variance": float(context["metric_value"].var(ddof=0)),
        "score_max": float(target_slice[score_column].max()),
        "context_score_max": float(context[score_column].max()),
        "threshold": threshold,
        "score_to_threshold_ratio": _safe_divide(
            float(target_slice[score_column].max()), threshold
        ),
        "source_item": item,
    }


def _noisy_normal_case(predictions: pd.DataFrame, *, context_points: int) -> dict[str, Any] | None:
    candidates = predictions[
        predictions["dense_autoencoder_prediction"].eq(1) & predictions["y_true"].eq(0)
    ].copy()
    if candidates.empty:
        return None
    candidates["score_margin"] = (
        candidates["dense_autoencoder_score"] - candidates["dense_autoencoder_threshold"]
    )
    row = candidates.sort_values("score_margin", ascending=False).iloc[0]
    return _case_context(
        predictions=predictions,
        model_name="dense_autoencoder",
        case_type="noisy_normal_period",
        item={
            "entity_id": row["entity_id"],
            "timestamp": str(row["timestamp"]),
            "metric_value": float(row["metric_value"]),
            "score": float(row["dense_autoencoder_score"]),
        },
        context_points=context_points,
    )


def _plot_failure_case(
    output_dir: Path,
    predictions: pd.DataFrame,
    case: dict[str, Any],
    *,
    model_name: str,
) -> None:
    entity_id = str(case["entity_id"])
    group = predictions[predictions["entity_id"].eq(entity_id)].sort_values("timestamp")
    timestamps = pd.to_datetime(group["timestamp"], errors="coerce")
    start = pd.to_datetime(case["start_timestamp"], errors="coerce")
    end = pd.to_datetime(case["end_timestamp"], errors="coerce")
    frame = group[(timestamps >= start) & (timestamps <= end)]
    if frame.empty:
        return
    score_column = f"{model_name}_score"
    threshold_column = f"{model_name}_threshold"
    prediction_column = f"{model_name}_prediction"
    figure, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(frame["timestamp"], frame["metric_value"], color="#334155", linewidth=1.0)
    positives = frame[frame["y_true"].eq(1)]
    detections = frame[frame[prediction_column].eq(1)]
    if not positives.empty:
        axes[0].scatter(
            positives["timestamp"],
            positives["metric_value"],
            color="#f59e0b",
            s=16,
            label="label",
        )
    if not detections.empty:
        axes[0].scatter(
            detections["timestamp"],
            detections["metric_value"],
            facecolors="none",
            edgecolors="#dc2626",
            s=24,
            label="detection",
        )
    axes[0].set_title(f"{case['case_type']}: {entity_id}")
    axes[0].set_ylabel("metric")
    axes[1].plot(frame["timestamp"], frame[score_column], color="#2563eb", linewidth=1.0)
    axes[1].axhline(
        float(frame[threshold_column].iloc[0]),
        color="#dc2626",
        linestyle="--",
        linewidth=1.0,
    )
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("time")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        if labels:
            axis.legend(handles, labels, loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()
    filename = f"failure_{_safe_filename(str(case['case_type']))}_{_safe_filename(entity_id)}.png"
    figure.savefig(output_dir / "figures" / filename)
    plt.close(figure)


def _select_labeled_series_for_window_sensitivity(
    series_items: list[NabSeries],
    base_config: NabDeepExperimentConfig,
    *,
    max_series: int,
) -> list[NabSeries]:
    selected: list[NabSeries] = []
    for series in sorted(series_items, key=lambda item: item.entity_id):
        prepared = _prepare_series(series, base_config)
        if int(prepared.test_eval["y_true"].sum()) > 0:
            selected.append(series)
        if len(selected) >= max_series:
            break
    return selected


def _window_variant_config(
    base_config: NabDeepExperimentConfig,
    window_size: int,
    config: NabRobustnessConfig,
) -> NabDeepExperimentConfig:
    return replace(
        base_config,
        windowing=replace(base_config.windowing, sequence_length=window_size),
        dense_autoencoder_config=replace(
            base_config.dense_autoencoder_config,
            epochs=config.window_sensitivity_epochs,
            patience=min(
                base_config.dense_autoencoder_config.patience, config.window_sensitivity_epochs
            ),
        ),
        lstm_autoencoder_config=replace(
            base_config.lstm_autoencoder_config,
            epochs=config.window_sensitivity_epochs,
            patience=min(
                base_config.lstm_autoencoder_config.patience, config.window_sensitivity_epochs
            ),
        ),
        ablation=replace(base_config.ablation, enabled=False),
    )


def _plot_series_classification(run_dir: Path, series_analysis: pd.DataFrame) -> None:
    counts = (
        series_analysis["difficulty_class"]
        .value_counts()
        .reindex(
            ["easy", "moderate", "difficult", "complete_failure"],
            fill_value=0,
        )
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(counts.index, counts.values, color=["#0f766e", "#2563eb", "#f59e0b", "#dc2626"])
    axis.set_title("NAB Series Difficulty Classes")
    axis.set_xlabel("class")
    axis.set_ylabel("series")
    figure.tight_layout()
    figure.savefig(run_dir / "figures" / "01_series_difficulty_classes.png")
    plt.close(figure)


def _plot_threshold_sensitivity(run_dir: Path, threshold_global: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    x_labels = [f"{row.model}\n{row.threshold_strategy}" for row in threshold_global.itertuples()]
    axis.bar(x_labels, threshold_global["f1"], color="#2563eb")
    axis.set_title("Threshold Sensitivity: Global F1")
    axis.set_ylabel("F1")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(run_dir / "figures" / "02_threshold_sensitivity_f1.png")
    plt.close(figure)


def _plot_precision_recall_tradeoff(run_dir: Path, tradeoff: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    for model_name, group in tradeoff.groupby("model", sort=True):
        axis.plot(group["recall"], group["precision"], label=model_name)
    axis.set_title("Diagnostic Precision-Recall Tradeoff")
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(run_dir / "figures" / "03_precision_recall_tradeoff.png")
    plt.close(figure)


def _plot_complementarity(run_dir: Path, summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    labels = ["both", "isolation_only", "dense_only"]
    values = [int(row[label]) for label in labels]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color=["#64748b", "#dc2626", "#0891b2"])
    axis.set_title("IF vs Dense AE Detection Overlap")
    axis.set_ylabel("test points")
    figure.tight_layout()
    figure.savefig(run_dir / "figures" / "04_if_dense_overlap.png")
    plt.close(figure)


def _plot_window_sensitivity(run_dir: Path, window_global: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    for model_name, group in window_global.groupby("model", sort=True):
        ordered = group.sort_values("window_size")
        axis.plot(
            ordered["window_size"],
            ordered["f1"],
            marker="o",
            label=model_name,
        )
    axis.set_title("Neural Window-Size Sensitivity")
    axis.set_xlabel("window size")
    axis.set_ylabel("F1")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(run_dir / "figures" / "05_window_sensitivity_f1.png")
    plt.close(figure)


def _add_missing_model_metrics(row: dict[str, Any], model: ModelName) -> None:
    for suffix in (
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "false_positive_rate",
        "detected_windows",
        "missed_windows",
        "median_delay_steps",
    ):
        row[f"{model}_{suffix}"] = np.nan


def _best_threshold_rows(threshold_global: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _model_name, group in threshold_global.groupby("model", sort=True):
        best = group.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
        rows.append(cast(dict[str, Any], best.to_dict()))
    return rows


def _best_window_rows(window_global: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _model_name, group in window_global.groupby("model", sort=True):
        best = group.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
        rows.append(cast(dict[str, Any], best.to_dict()))
    return rows


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def _value_counts(values: pd.Series[Any]) -> dict[str, int]:
    counts = values.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _finite_scores(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("score array must contain at least one finite value")
    return scores


def _percentile_threshold(scores: np.ndarray, percentile: float) -> float:
    return float(np.percentile(_finite_scores(scores), percentile))


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _float_value(value: Any) -> float:
    maybe = _maybe_float(value)
    return 0.0 if maybe is None else maybe


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


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
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


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


def _write_run_report(run_dir: Path, smd_profile_dir: Path, summary: dict[str, Any]) -> None:
    classes = summary["series_classification_counts"]
    lines = [
        "# Phase 5 NAB Robustness Analysis",
        "",
        f"Generated at: `{summary['generated_at']}`",
        f"Source Phase 4 run: `{summary['source_phase4_run_dir']}`",
        "",
        "## Series Difficulty",
        "",
        "| Class | Series |",
        "| --- | ---: |",
    ]
    for label in ("easy", "moderate", "difficult", "complete_failure"):
        lines.append(f"| {label} | {classes.get(label, 0)} |")
    lines.extend(
        [
            "",
            "## Threshold Sensitivity",
            "",
            "| Model | Best strategy | F1 | Precision | Recall | FPR |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["threshold_best_by_f1"]:
        lines.append(
            "| {model} | {strategy} | {f1:.4f} | {precision:.4f} | {recall:.4f} | "
            "{fpr:.4f} |".format(
                model=row["model"],
                strategy=row["threshold_strategy"],
                f1=row["f1"],
                precision=row["precision"],
                recall=row["recall"],
                fpr=row["false_positive_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Window Sensitivity",
            "",
            "| Model | Best window | F1 | Precision | Recall |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["window_sensitivity_best_by_f1"]:
        lines.append(
            "| {model} | {window} | {f1:.4f} | {precision:.4f} | {recall:.4f} |".format(
                model=row["model"],
                window=int(row["window_size"]),
                f1=row["f1"],
                precision=row["precision"],
                recall=row["recall"],
            )
        )
    lines.extend(
        [
            "",
            "## Official NAB Scoring",
            "",
            summary["official_nab_scoring"]["phase5_decision"],
            "",
            "## SMD Profile",
            "",
            f"SMD profile artifacts: `{smd_profile_dir.as_posix()}`",
        ]
    )
    (run_dir / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
