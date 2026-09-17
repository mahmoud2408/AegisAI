# Dataset Audit

Latest local audit command:

```powershell
python scripts/datasets/audit_datasets.py --pretty
```

Audit artifact: `data/manifests/dataset_audit.json`.

The artifact contains exact local file paths, sizes, formats, and per-dataset profiles for the current machine. It is ignored by Git because it describes local raw data.

## Data Lake Inventory

| Stage | Exists | Files | Size | Status |
| --- | --- | ---: | ---: | --- |
| `data/raw` | yes | 248 | 895.0 MiB | Contains downloaded public data plus manual placeholders. |
| `data/interim` | yes | 1 | 1 B | Placeholder only. |
| `data/processed` | yes | 1 | 1 B | Placeholder only. |
| `data/synthetic` | yes | 1 | 1 B | Placeholder only. |
| `data/manifests` | yes | 3 | 206.8 KiB | Contains `.gitkeep`, download manifest, and audit manifest. |

Additional empty raw placeholders exist for `data/raw/smap` and `data/raw/msl`; the combined configured dataset is currently `data/raw/smap_msl`.

## Dataset Inventory

| Dataset | Path | Health | Files | Size | Rows / Dimensions | Labels |
| --- | --- | --- | ---: | ---: | --- | --- |
| NAB | `data/raw/nab` | READY | 71 | 9.2 MiB | 58 series, 365,558 rows, 2 columns | 120 point labels, 116 windows |
| SMD | `data/raw/smd` | READY | 113 | 463.5 MiB | 28 machines, 1,416,825 rows, 38 metrics | 708,420 test labels, 29,444 positives |
| SMAP/MSL | `data/raw/smap_msl` | INCOMPLETE | 1 | 3.9 KiB | 82 label rows only | 105 anomaly intervals, no telemetry |
| LogHub HDFS | `data/raw/loghub/hdfs` | READY | 5 | 692.9 KiB | 2,000 structured rows, 14 templates | No anomaly label column found |
| LogHub BGL | `data/raw/loghub/bgl` | READY | 5 | 764.5 KiB | 2,000 structured rows, 120 templates | Row-level label column |
| LogHub OpenStack | `data/raw/loghub/openstack` | READY | 4 | 1.3 MiB | 2,000 structured rows, 43 templates | No anomaly label column found |
| LogHub Hadoop | `data/raw/loghub/hadoop` | READY | 4 | 914.4 KiB | 2,000 structured rows, 114 templates | No anomaly label column found |
| LogHub Spark | `data/raw/loghub/spark` | READY | 4 | 492.5 KiB | 2,000 structured rows, 36 templates | No anomaly label column found |
| LogHub Zookeeper | `data/raw/loghub/zookeeper` | READY | 4 | 640.8 KiB | 2,000 structured rows, 50 templates | No anomaly label column found |
| OpenTelemetry resources | `data/raw/opentelemetry` | REFERENCE_READY | 11 | 127.8 KiB | No static telemetry rows | No labels |
| MetroPT-3 | `data/raw/metropt` | READY | 3 | 416.5 MiB | 1,516,948 rows, 17 columns | No machine-readable label column |
| AI4I 2020 | `data/raw/ai4i` | READY | 2 | 1019.7 KiB | 10,000 rows, 14 columns | 339 machine failures |
| SWaT | `data/raw/swat` | MANUAL_ACCESS | 1 | 657 B | Placeholder only | Not inspectable |
| WADI | `data/raw/wadi` | MANUAL_ACCESS | 1 | 657 B | Placeholder only | Not inspectable |
| CIC-IDS2017 | `data/raw/cicids2017` | MANUAL_DOWNLOAD | 1 | 760 B | Placeholder only | Not inspectable |

## Dataset Characterization

| Dataset | Data Type | Domain | Size | Dimensions | Labels | Temporal | Main Use | Priority |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| NAB | Univariate time series | Cloud, traffic, ads, social, artificial | 9.2 MiB | 58 series x 2 columns | Point labels and windows | Timestamped, mixed frequencies | Univariate anomaly benchmark | P0 |
| SMD | Multivariate time series | Server telemetry | 463.5 MiB | 28 machines x 38 metrics | Point-wise test labels | Row-index sequence, no timestamp | Multivariate anomaly benchmark | P0 |
| SMAP/MSL | Label metadata | Spacecraft telemetry | 3.9 KiB | 82 channels in label file | Interval metadata only | No local telemetry | Future benchmark if completed | P3 |
| LogHub HDFS | Structured logs | Distributed filesystem | 692.9 KiB | 2,000 rows, 9 structured columns | None local | Event timestamps, irregular | Log parsing/templates | P1 |
| LogHub BGL | Structured logs | HPC system logs | 764.5 KiB | 2,000 rows, 13 structured columns | Row labels | Event timestamps, irregular | Labeled log anomaly task | P1 |
| LogHub OpenStack | Structured logs | Cloud infrastructure | 1.3 MiB | 2,000 rows, 11 structured columns | None local | Event timestamps, irregular | Request-aware log features | P1 |
| LogHub Hadoop | Structured logs | Hadoop jobs | 914.4 KiB | 2,000 rows, 9 structured columns | None local | Event timestamps, bursty | Log event sequences | P1 |
| LogHub Spark | Structured logs | Spark jobs | 492.5 KiB | 2,000 rows, 8 structured columns | None local | 31-second event slice | Parser smoke tests | P2 |
| LogHub Zookeeper | Structured logs | Coordination service | 640.8 KiB | 2,000 rows, 10 structured columns | None local | Event timestamps, irregular | Service log sequences | P1 |
| OpenTelemetry | Config/reference files | Observability demo | 127.8 KiB | 11 files, no telemetry rows | None | Not a static dataset | Synthetic generator reference | P1 |
| MetroPT-3 | Multivariate time series | Air-compressor maintenance | 416.5 MiB | 1,516,948 rows, 17 columns | None in CSV | Timestamped, mostly 10 seconds | Forecasting and unsupervised detection | P0 |
| AI4I 2020 | Tabular classification | Predictive maintenance | 1019.7 KiB | 10,000 rows, 14 columns | Failure and failure modes | No timestamp | Failure prediction and XAI | P0 |
| SWaT | Industrial-control time series | Water treatment | 657 B | Placeholder only | Not inspectable | Not inspectable | Future ICS benchmark | P2/P3 |
| WADI | Industrial-control time series | Water distribution | 657 B | Placeholder only | Not inspectable | Not inspectable | Future ICS benchmark | P2/P3 |
| CIC-IDS2017 | Network traffic | Intrusion detection | 760 B | Placeholder only | Not inspectable | Not inspectable | Future security extension | P3 |

## NAB Audit

Files:

- 58 CSV time-series files under `data/raw/nab/data`.
- 10 JSON label files under `data/raw/nab/labels`.
- README/license text files.

Measured profile:

- Rows: 365,558.
- Columns: `timestamp`, `value`.
- Missing values: 0.
- Duplicate rows: 17.
- Duplicate timestamps: 49 across all series.
- Constant value series: 1.
- Labeled series: 52.
- Point labels: 120.
- Scoring windows: 116.
- Categories: artificialNoAnomaly 5, artificialWithAnomaly 6, realAWSCloudwatch 17, realAdExchange 6, realKnownCause 7, realTraffic 7, realTweets 10.
- Temporal coverage: min `2011-07-01 00:00:01`, max `2015-09-17 17:10:00`.
- Dominant cadences: 300 seconds, 3,600 seconds, 1,800 seconds, 600 seconds.
- Regular series files: 33 of 58.

Use:

- Best first anomaly-detection benchmark.
- Good for NAB scoring and detection-delay metrics.
- Useful for univariate forecasting baselines.

Limits:

- Many domains are not infrastructure incidents.
- Mixed cadences require per-series processing.
- Labels are anomaly timestamps/windows, not root-cause labels.

## SMD Audit

Files:

- 28 train matrices in `data/raw/smd/train`.
- 28 test matrices in `data/raw/smd/test`.
- 28 test label vectors in `data/raw/smd/test_label`.
- 28 interpretation-label files in `data/raw/smd/interpretation_label`.
- 1 license file.

Measured profile:

- Machines: 28.
- Train rows: 708,405.
- Test rows: 708,420.
- Total telemetry rows: 1,416,825.
- Columns per telemetry row: 38.
- Missing values: 0.
- Duplicate rows: 0.
- Test label rows: 708,420.
- Positive test labels: 29,444.
- Positive label rate: 4.156%.
- Test/label row-count mismatches: none.
- Interpretation intervals: 327.
- Timestamp column: absent; row index is the only temporal axis.

Use:

- Core multivariate anomaly benchmark.
- Good for autoencoder/LSTM autoencoder later.
- Good for leakage-safe incident-window prediction because labels are point-wise.
- Interpretation labels can support evidence ranking by metric index.

Limits:

- Metric names are anonymous.
- No real timestamps or sampling rate in local files.
- Train/test split should be respected; do not random-split the combined dataset.

## SMAP/MSL Audit

Files:

- `data/raw/smap_msl/labeled_anomalies.csv` only.
- Expected `data.zip` telemetry bundle is absent.

Measured profile:

- Label rows/channels: 82.
- SMAP channels: 55.
- MSL channels: 27.
- Anomaly sequences: 105.
- Anomaly classes observed: contextual 43, point 62.
- Missing values: 0.
- Duplicate rows: 0.
- Telemetry bundle present: false.

Use:

- Not usable for model training or evaluation yet.
- Keep label metadata as provenance and as a future adapter target.

Limits:

- Labels cannot be joined to observations because observations are missing.

## LogHub Audit

All six local LogHub datasets are small 2k samples with raw logs, structured CSVs, and template CSVs.

| Dataset | Structured Columns | Missing | Duplicate Rows | Duplicate Timestamps | Level Counts | Templates |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| HDFS | LineId, Date, Time, Pid, Level, Component, Content, EventId, EventTemplate | 0 | 0 | 117 | INFO 1920, WARN 80 | 14 |
| BGL | LineId, Label, Timestamp, Date, Node, Time, NodeRepeat, Type, Component, Level, Content, EventId, EventTemplate | 0 | 0 | 17 | INFO 1597, FATAL 347, ERROR 41, WARNING 8, SEVERE 7 | 120 |
| OpenStack | LineId, Logrecord, Date, Time, Pid, Level, Component, ADDR, Content, EventId, EventTemplate | 0 | 0 | 67 | INFO 1969, WARNING 31 | 43 |
| Hadoop | LineId, Date, Time, Level, Process, Component, Content, EventId, EventTemplate | 0 | 0 | 888 | INFO 1040, WARN 808, ERROR 150, FATAL 2 | 114 |
| Spark | LineId, Date, Time, Level, Component, Content, EventId, EventTemplate | 0 | 0 | 1980 | INFO 2000 | 36 |
| Zookeeper | LineId, Date, Time, Level, Node, Component, Id, Content, EventId, EventTemplate | 0 | 0 | 57 | WARN 1318, INFO 669, ERROR 13 | 50 |

BGL label profile:

- `-`: 1,857 rows.
- Non-normal labels: 143 rows across APP and KERN categories.

Use:

- HDFS/OpenStack/Hadoop/Spark/Zookeeper: parser evaluation, event-template features, log sequence representations.
- BGL: labeled log anomaly classification or sequence anomaly detection.
- OpenStack: future request-aware log grouping using `ADDR`.

Limits:

- These are small samples, not full production-scale LogHub releases.
- Most local samples do not include anomaly labels.
- Template files must not be treated as anomaly labels.

## MetroPT-3 Audit

Files:

- Original UCI ZIP archive.
- `MetroPT3(AirCompressor).csv`.
- `Data Description_Metro.pdf`.

Measured profile:

- Rows: 1,516,948.
- Columns: 17.
- Missing values: 0.
- Duplicate rows: 0.
- Duplicate timestamps: 0.
- Temporal coverage: `2020-02-01 00:00:00` to `2020-09-01 03:59:50`.
- Median cadence: 10 seconds.
- Dominant deltas: 10 seconds, 9 seconds, 12 seconds, 13 seconds, 11 seconds.
- Gaps greater than 1.5x median delta: 363.

Columns:

- Identifier/index: `unnamed_index`.
- Timestamp: `timestamp`.
- Continuous numeric sensors: `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, `Motor_current`.
- Binary state columns: `COMP`, `DV_eletric`, `Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Caudal_impulses`.

Selected numeric ranges:

- `TP2`: min -0.032, max 10.676, mean 1.368.
- `TP3`: min 0.730, max 10.302, mean 8.985.
- `H1`: min -0.036, max 10.288, mean 7.568.
- `DV_pressure`: min -0.032, max 9.844, mean 0.056.
- `Oil_temperature`: min 15.400, max 89.050, mean 62.644.
- `Motor_current`: min 0.020, max 9.295, mean 2.050.

Use:

- Best high-volume forecasting dataset currently available.
- Good for unsupervised/semi-supervised predictive-maintenance anomaly detection.
- Supports a documented Phase 7 early-warning case study after curating failure intervals from the source report.
- Good for operating-state segmentation because binary state columns are present.

Limits:

- No machine-readable failure label column is present in the inspected CSV.
- Failure-prediction claims must be limited to documented report-derived event labels; do not treat the CSV as natively labeled.
- Sampling is near-regular, not perfectly regular.

## AI4I 2020 Audit

Files:

- Original UCI ZIP archive.
- `ai4i2020.csv`.

Measured profile:

- Rows: 10,000.
- Columns: 14.
- Missing values: 0.
- Duplicate rows: 0.
- Identifiers: `UDI`, `Product ID`.
- Categorical columns: `Type`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
- Numeric feature columns: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
- No timestamp column.

Label counts:

- `Machine failure`: 339 positive, 9,661 negative.
- `TWF`: 46 positive.
- `HDF`: 115 positive.
- `PWF`: 95 positive.
- `OSF`: 98 positive.
- `RNF`: 19 positive.

Selected numeric ranges:

- Air temperature: min 295.3 K, max 304.5 K, mean 300.005 K.
- Process temperature: min 305.7 K, max 313.8 K, mean 310.006 K.
- Rotational speed: min 1,168 rpm, max 2,886 rpm, mean 1,538.776 rpm.
- Torque: min 3.8 Nm, max 76.6 Nm, mean 39.987 Nm.
- Tool wear: min 0 min, max 253 min, mean 107.951 min.

Use:

- Compact supervised predictive-maintenance baseline.
- Good for probability calibration and SHAP-style explainability.

Limits:

- Synthetic dataset.
- No chronological information; cannot support temporal incident forecasting claims.
- `UDI` and `Product ID` are identifiers and should not be predictive features.

## OpenTelemetry Resources Audit

Files:

- 11 reference files, including generator, collector, Prometheus, Grafana, and trace-test configuration files.

Measured profile:

- Static telemetry rows: 0.
- Labels: none.

Use:

- Design input for AegisAI synthetic telemetry/log/trace generator.
- Useful for validating names, collector config patterns, and trace/log export assumptions.

Limits:

- Not a benchmark.
- Cannot be used to claim detection, prediction, or RCA performance.

## Manual Dataset Placeholders

SWaT, WADI, and CIC-IDS2017 are not locally installed beyond README placeholders.

Use:

- Keep documented as future extensions.
- Do not include in current benchmark claims.

Limits:

- No rows, columns, labels, timestamps, or quality statistics can be measured yet.

## Quality Findings

| Dataset | Quality Score | Key Issues |
| --- | ---: | --- |
| NAB | 91 | Mixed frequencies, 17 duplicate rows, 49 duplicate timestamps, 1 constant series. |
| SMD | 88 | Anonymous metrics, row-index time only, some constant dimensions by machine. |
| MetroPT-3 | 79 | No CSV failure labels, near-regular cadence with 363 larger gaps. |
| AI4I 2020 | 85 | Synthetic, non-temporal, strong label imbalance. |
| LogHub BGL | 84 | Small sample, irregular events, imbalanced labels. |
| LogHub HDFS | 75 | No local anomaly labels, irregular multi-year sample timestamps. |
| LogHub OpenStack | 77 | No local labels, request identifiers need parsing from `ADDR`. |
| LogHub Hadoop | 74 | No local labels, many same-second/millisecond duplicate timestamps. |
| LogHub Zookeeper | 73 | No local labels, irregular event bursts. |
| LogHub Spark | 66 | Only 31 seconds of events and all rows are `INFO`. |
| OpenTelemetry resources | 48 | Reference files only, no measured telemetry rows. |
| SMAP/MSL | 30 | Telemetry bundle missing. |
| SWaT/WADI/CIC-IDS2017 | 10 | Manual placeholders only. |

## Health Summary

Ready for immediate benchmark work:

- NAB
- SMD
- MetroPT-3
- AI4I 2020
- LogHub BGL
- LogHub HDFS/OpenStack/Hadoop/Spark/Zookeeper for parser/log-feature work

Usable only as reference or future work:

- OpenTelemetry resources
- SMAP/MSL label metadata

Not usable until installed:

- SWaT
- WADI
- CIC-IDS2017
