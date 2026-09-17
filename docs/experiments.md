# Experiments

Phase status: MetroPT time-series forecasting completed.

This file will track reproducible experiments only. Do not add manually invented scores.

## Experiment Record Template

| Field | Value |
| --- | --- |
| Experiment ID | |
| Date | |
| Dataset | |
| Dataset version or commit | |
| Split policy | |
| Model | |
| Parameters | |
| Metrics artifact | |
| Model artifact | |
| Reproduction command | |
| Notes | |

## Planned Benchmark Families

- NAB univariate anomaly detection
- SMD multivariate anomaly detection
- AI4I and MetroPT failure prediction
- Synthetic scenario anomaly detection and incident prediction
- Forecasting backtests
- RAG retrieval evaluation
- Agent investigation ablations

## Results

### NAB Anomaly Baseline

| Field | Value |
| --- | --- |
| Experiment ID | `run_phase3_baseline` |
| Date | 2026-09-03 |
| Dataset | Numenta Anomaly Benchmark (processed local Phase 2 artifact) |
| Dataset version or commit | Local processed metrics/labels under `data/processed`; raw datasets are not committed |
| Split policy | Per-series chronological 50% train, 20% validation, 30% test |
| Models | Rolling z-score baseline; Isolation Forest |
| Parameters | Isolation Forest: `n_estimators=200`, `max_samples=auto`, `contamination=auto`, `random_state=42`; threshold = validation score 99th percentile |
| Metrics artifact | `experiments/anomaly/nab/isolation_forest/run_phase3_baseline/metrics.json` |
| Model artifact | `experiments/anomaly/nab/isolation_forest/run_phase3_baseline/models/*.joblib` |
| Reproduction command | `.\.venv\Scripts\python.exe scripts\experiments\run_nab_baseline.py --run-id run_phase3_baseline` |
| Notes | Standard temporal point/window metrics only; official NAB normalized score is not reported |

Measured global test metrics:

| Model | Precision | Recall | F1 | PR-AUC | FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rolling z-score | 0.0589 | 0.0139 | 0.0225 | 0.1185 | 0.0268 |
| Isolation Forest | 0.2928 | 0.1316 | 0.1815 | 0.2147 | 0.0384 |

Detailed write-up: `docs/experiments/nab_baseline.md`

### NAB Deep Anomaly Detection

| Field | Value |
| --- | --- |
| Experiment ID | `run_phase4_deep_anomaly` |
| Date | 2026-09-04 |
| Dataset | Numenta Anomaly Benchmark (processed local Phase 2 artifact) |
| Dataset version or commit | Local processed metrics/labels under `data/processed`; raw datasets are not committed |
| Split policy | Per-series chronological 50% train, 20% validation, 30% test |
| Models | Rolling z-score, Isolation Forest, Dense Autoencoder, LSTM Autoencoder |
| Parameters | Window length 32, stride 1, train-percentile threshold 99, train-only standard scaling |
| Metrics artifact | `experiments/anomaly/nab/deep_autoencoders/run_phase4_deep_anomaly/metrics.json` |
| Model artifact | `experiments/anomaly/nab/deep_autoencoders/run_phase4_deep_anomaly/models/` |
| Reproduction command | `.\.venv\Scripts\python.exe scripts\experiments\run_nab_deep_anomaly.py --run-id run_phase4_deep_anomaly` |
| Notes | Uses common window-ending test timestamps for all four models; official NAB normalized score is not reported |

Measured global test metrics:

| Model | Precision | Recall | F1 | PR-AUC | FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rolling z-score | 0.0660 | 0.0142 | 0.0233 | 0.1175 | 0.0239 |
| Isolation Forest | 0.2511 | 0.1397 | 0.1795 | 0.2093 | 0.0498 |
| Dense Autoencoder | 0.2964 | 0.2718 | 0.2835 | 0.1796 | 0.0771 |
| LSTM Autoencoder | 0.2523 | 0.2245 | 0.2376 | 0.1913 | 0.0795 |

Detailed write-up: `docs/experiments/nab_deep_anomaly.md`
Label audit: `docs/experiments/nab_label_alignment.md`

### NAB Robustness and SMD Transition Analysis

| Field | Value |
| --- | --- |
| Experiment ID | `run_phase5_nab_robustness_final` |
| Date | 2026-09-04 |
| Dataset | Numenta Anomaly Benchmark plus SMD profile inspection |
| Dataset version or commit | Local processed NAB metrics/labels and local raw/processed SMD artifacts; raw datasets are not committed |
| Split policy | Reuses Phase 4 NAB chronological split; SMD is profiled only |
| Models | No new production model; saved IF/Dense checkpoints for threshold diagnostics; bounded Dense/LSTM window-size diagnostic |
| Parameters | IF/Dense thresholds: train-p99, validation-p99, train-MAD; neural window diagnostic: 16, 32, 64 on six labeled NAB series, two epochs |
| Metrics artifact | `experiments/anomaly/nab/robustness/run_phase5_nab_robustness_final/summary.json` |
| Model artifact | Reuses `experiments/anomaly/nab/deep_autoencoders/run_phase4_deep_anomaly/models/` |
| Reproduction command | `.\.venv\Scripts\python.exe scripts\experiments\analyze_nab_robustness.py --run-id run_phase5_nab_robustness_final --smd-run-id run_phase5_smd_profile_final` |
| Notes | Official NAB normalized score is explicitly not reported; local NAB mirror lacks the official scorer/application-profile implementation |

Measured difficulty classification:

| Class | Series |
| --- | ---: |
| Easy | 10 |
| Moderate | 31 |
| Difficult | 12 |
| Complete failure | 5 |

Best measured threshold strategies:

| Model | Strategy | Precision | Recall | F1 | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense Autoencoder | validation-p99 | 0.3425 | 0.2901 | 0.3141 | 0.0665 |
| Isolation Forest | train-p99 | 0.2511 | 0.1397 | 0.1795 | 0.0498 |

SMD profile summary:

| Property | Value |
| --- | ---: |
| Machines | 28 |
| Metrics per machine | 38 |
| Train rows | 708,405 |
| Test rows | 708,420 |
| Positive test labels | 29,444 |
| Positive label rate | 0.0416 |
| Anomaly segments | 327 |

Detailed write-ups:

- `docs/experiments/nab_error_analysis.md`
- `docs/experiments/nab_robustness.md`
- `docs/experiments/smd_plan.md`

### SMD Multivariate Anomaly Detection

| Field | Value |
| --- | --- |
| Experiment ID | `phase6_smd_multivariate_20260911_eps1e3` |
| Date | 2026-09-11 |
| Dataset | Server Machine Dataset |
| Dataset version or commit | Local raw SMD matrices under `data/raw/smd`; raw datasets are not committed |
| Split policy | Source train split chronologically into 80% fit and 20% validation; source test labels held out for final evaluation |
| Models | Per-feature z-score, point-wise multivariate Isolation Forest, Dense Autoencoder, LSTM Autoencoder |
| Parameters | 38 metrics, window size 64 for neural models, stride 1, train/validation percentile thresholds at p99, scaler epsilon `1e-3` |
| Metrics artifact | `experiments/anomaly/smd/phase6_smd_multivariate_20260911_eps1e3/metrics.json` |
| Model artifact | `experiments/anomaly/smd/phase6_smd_multivariate_20260911_eps1e3/models/` |
| Reproduction command | `.\.venv\Scripts\python.exe scripts\experiments\run_smd_multivariate.py --run-id phase6_smd_multivariate_20260911_eps1e3` |
| Notes | SMD has no native timestamps; detection delay is in sequence steps. NAB score is not applicable to SMD. |

Measured global test metrics:

| Model | Threshold | Precision | Recall | F1 | PR-AUC | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Isolation Forest | train_percentile | 0.1622 | 0.3636 | 0.2243 | 0.2022 | 0.0817 |
| Isolation Forest | validation_percentile | 0.1471 | 0.3508 | 0.2072 | 0.2022 | 0.0885 |
| Per-feature z-score | validation_percentile | 0.1147 | 0.4944 | 0.1861 | 0.1002 | 0.1660 |
| Dense Autoencoder | train_percentile | 0.0960 | 0.5644 | 0.1641 | 0.1131 | 0.2310 |
| Per-feature z-score | train_percentile | 0.0960 | 0.5390 | 0.1630 | 0.1002 | 0.2207 |
| LSTM Autoencoder | train_percentile | 0.0936 | 0.5577 | 0.1603 | 0.1097 | 0.2349 |
| LSTM Autoencoder | validation_percentile | 0.0771 | 0.3620 | 0.1272 | 0.1097 | 0.1883 |
| Dense Autoencoder | validation_percentile | 0.0749 | 0.4170 | 0.1270 | 0.1131 | 0.2239 |

Detailed write-up: `docs/experiments/smd_multivariate.md`

### Failure Prediction and Predictive Maintenance

| Field | Value |
| --- | --- |
| Experiment ID | `phase7_failure_prediction_20260912` |
| Date | 2026-09-12 |
| Dataset | AI4I 2020 and MetroPT-3 |
| Dataset version or commit | Local raw AI4I and MetroPT files under `data/raw`; raw datasets are not committed |
| Split policy | AI4I stratified train/validation/test; MetroPT event-aware chronological train/validation/test |
| Models | Logistic Regression, Random Forest, XGBoost |
| Parameters | AI4I features: five numeric process variables plus `Type`; MetroPT horizon: 6 hours; MetroPT stride: 6 raw rows; threshold grid: 0.1, 0.2, 0.3, 0.5, 0.7 |
| Metrics artifact | `experiments/prediction/failure_prediction/phase7_failure_prediction_20260912/metrics.json` |
| Model artifact | `experiments/prediction/failure_prediction/phase7_failure_prediction_20260912/*/models/` |
| Reproduction command | `.\.venv\Scripts\python.exe scripts\experiments\run_failure_prediction.py --run-id phase7_failure_prediction_20260912` |
| Notes | MetroPT CSV has no native label column; failure intervals are curated from the local source report and used only as documented case-study labels. |

Measured AI4I test metrics:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.2782 | 0.7255 | 0.4022 | 0.9101 | 0.4219 | 0.0662 |
| Random Forest | 0.6596 | 0.6078 | 0.6327 | 0.9674 | 0.6757 | 0.0110 |
| XGBoost | 0.5606 | 0.7255 | 0.6325 | 0.9659 | 0.7450 | 0.0200 |

Measured MetroPT test metrics:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Warning coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.0000 | 0.0000 | 0.0000 | 0.1282 | 0.0020 | 0.0507 | 0.0000 |
| Random Forest | 0.0000 | 0.0000 | 0.0000 | 0.2509 | 0.0031 | 0.0142 | 0.0000 |
| XGBoost | 0.0000 | 0.0000 | 0.0000 | 0.1795 | 0.0035 | 0.0126 | 0.0000 |

The AI4I best test F1 is Random Forest. MetroPT has no deployable winner in this run: every classical model missed the held-out July failure under validation-selected thresholds.

Detailed write-ups:

- `docs/experiments/ai4i_failure_prediction.md`
- `docs/experiments/metropt_failure_prediction.md`
- `docs/experiments/lead_time_analysis.md`

### MetroPT Failure Prediction Robustness and Target Validation

| Field | Value |
| --- | --- |
| Experiment ID | `phase7b_metropt_robustness_20260914` |
| Date | 2026-09-14 |
| Dataset | MetroPT-3 |
| Dataset version or commit | Local raw MetroPT files under `data/raw/metropt`; raw datasets are not committed |
| Split policy | Event-aware chronological train/validation/test with active-failure rows excluded |
| Models | Logistic Regression, Random Forest, XGBoost |
| Parameters | Horizons: 0.5, 2.0, 6.0 hours; stride: 6 raw rows; validation-only threshold grid: 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5 |
| Metrics artifact | `experiments/prediction/metropt_robustness/phase7b_metropt_robustness_20260914/metrics.json` |
| Model artifact | `experiments/prediction/metropt_robustness/phase7b_metropt_robustness_20260914/horizon_*/models/` |
| Reproduction command | `.\.venv\Scripts\python.exe scripts\experiments\run_metropt_robustness.py --run-id phase7b_metropt_robustness_20260914` |
| Notes | The experiment validates the target construction, event table, threshold sensitivity, false-alarm burden, held-out July event behavior, distribution shift, and SHAP feature contributions. |

Failure event summary:

| Property | Value |
| --- | ---: |
| Curated failure events | 4 |
| Failure types | 1 |
| Held-out test failures | 1 |
| Overlapping failure intervals | 0 |
| Minimum gap between failure starts | 154.5 hours |

Best validation-selected candidate: `h6p0__random_forest`.

| Split | Precision | Recall | F1 | PR-AUC | Event detection | Median lead time hours | False alarms per day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.0694 | 0.4242 | 0.1193 | 0.0567 | 1.0000 | 5.9517 | 211.7724 |
| Test | 0.0000 | 0.0000 | 0.0000 | 0.0036 | 0.0000 | n/a | 377.2708 |

Final conclusion: Decision **B**. MetroPT supports a limited case-study early-warning target, but the supervised classifier is not deployable under the current labels. The validation-selected Random Forest missed the held-out July failure and produced a test false-positive rate of `0.4376`.

Detailed write-ups:

- `docs/experiments/metropt_failure_events.md`
- `docs/experiments/metropt_target_analysis.md`
- `docs/experiments/metropt_robustness.md`

### MetroPT Time-Series Forecasting

| Field | Value |
| --- | --- |
| Experiment ID | `phase8_metropt_forecasting_20260915` |
| Date | 2026-09-15 |
| Dataset | MetroPT-3 |
| Dataset version or commit | Local raw MetroPT file under `data/raw/metropt`; raw datasets are not committed |
| Split policy | Strict chronological train/validation/test with split-local contiguous windows |
| Models | Persistence, moving average, rolling linear trend, ridge autoregression, univariate LSTM, multivariate LSTM |
| Parameters | One-minute median resample; horizons 5, 15, 30 minutes; sequence length 60; stride 5; LSTM hidden size 32; 5 epochs with early stopping |
| Metrics artifact | `experiments/forecasting/metropt/phase8_metropt_forecasting_20260915/metrics.json` |
| Model artifact | `experiments/forecasting/metropt/phase8_metropt_forecasting_20260915/models/` |
| Reproduction command | `.\.venv\Scripts\python.exe scripts\experiments\run_metropt_forecasting.py --run-id phase8_metropt_forecasting_20260915` |
| Notes | Forecasts continuous telemetry only. Binary state variables are audited but not forecast. Transformer and uncertainty intervals are documented as exclusions. |

Data audit:

| Property | Value |
| --- | ---: |
| Raw rows | 1516948 |
| Raw median cadence | 10.0 seconds |
| Forecast cadence | 60.0 seconds |
| Resampled valid rows | 253371 |
| Segment count | 309 |
| Selected sensors | `TP2`, `TP3`, `H1`, `Reservoirs`, `Oil_temperature`, `Motor_current` |

Best validation-selected global candidate:

| Model | Horizon | Test MAE | Test RMSE | Test MAPE |
| --- | ---: | ---: | ---: | ---: |
| LSTM multivariate | 5 min | 1.2845 | 2.1687 | 10.6241 |

Best global test model by horizon:

| Horizon | Model | MAE | RMSE | MAPE |
| ---: | --- | ---: | ---: | ---: |
| 5 min | LSTM multivariate | 1.2845 | 2.1687 | 10.6241 |
| 15 min | Moving average | 1.4778 | 2.3374 | 12.8960 |
| 30 min | Moving average | 1.6402 | 2.5160 | 14.2721 |

Forecast-based degradation/risk signal summary:

| Metric | Value |
| --- | ---: |
| Mean precision | 0.3296 |
| Mean recall | 0.0698 |
| Mean F1 | 0.1030 |

The risk signal compares sampled forecasts against train-only non-failure normal ranges. It is a degradation evidence signal, not a failure-prediction model.

Detailed write-ups:

- `docs/experiments/metropt_forecasting.md`
- `docs/experiments/forecast_risk_signal.md`
- `docs/models/forecasting_baselines.md`
