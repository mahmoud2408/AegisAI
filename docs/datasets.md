# Datasets

Phase status: dataset bootstrap, audit, canonical adapters, and preprocessing are implemented. Raw and processed data remain local and ignored by Git.

Dataset definitions live in `config/datasets.yaml`. Dataset-specific notes live under `docs/datasets`.
The current measured audit is documented in `docs/dataset_audit.md`, and the recommended
dataset usage strategy is in `docs/data_strategy.md`.

## Sources

| Dataset | Planned use | Source |
| --- | --- | --- |
| Numenta Anomaly Benchmark (NAB) | Univariate streaming anomaly detection and NAB scoring | https://github.com/numenta/NAB |
| Server Machine Dataset (SMD) | Multivariate server telemetry anomaly detection | https://github.com/das2ar/OmniAnomaly |
| SMAP/MSL | Multivariate spacecraft telemetry anomaly detection | https://github.com/khundman/telemanom |
| LogHub | Log parsing, event templates, log anomaly/context signals | https://github.com/logpai/loghub |
| OpenTelemetry demo resources | Observability reference resources and future telemetry generation | https://github.com/openobserve/opentelemetry-demo-dataset |
| MetroPT-3 | Predictive maintenance and anomaly explanation | https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset |
| AI4I 2020 | Predictive maintenance classification and XAI practice | https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset |
| SWaT/WADI | Industrial-control anomaly detection | https://www.sutd.edu.sg/itrust/itrust-labs/datasets/ |
| CIC-IDS2017 | Cybersecurity anomaly detection | https://www.unb.ca/cic/datasets/ids-2017.html |

## Commands

```powershell
python scripts/datasets/download_all.py --list
python scripts/datasets/download_all.py --dataset public
python scripts/datasets/download_all.py --dataset nab
python scripts/datasets/download_all.py --dataset loghub
python scripts/datasets/validate_datasets.py
python scripts/datasets/audit_datasets.py --pretty
python scripts/data/profile_all.py
python scripts/data/preprocess_all.py --batch-size 25000
python scripts/data/validate_processed.py
```

The downloader updates `data/manifests/datasets_manifest.json` after each dataset attempt.
The audit command updates `data/manifests/dataset_audit.json` with local measured statistics.
The preprocessing command updates local profiles, lineage manifests, processed Parquet outputs,
and a preprocessing report.

## Canonical Preprocessing Status

Latest measured local run: 2026-09-03.

| Dataset | Status | Canonical output |
| --- | --- | --- |
| NAB | READY | 365,558 metric records; 236 label records |
| SMD | READY | 53,839,350 metric records; 708,747 label records |
| MetroPT-3 | READY | 22,754,220 metric records |
| AI4I 2020 | READY | 50,000 metric records; 10,000 failure records; 60,000 label records |
| SMAP | LABELS_ONLY | 69 label records |
| MSL | LABELS_ONLY | 36 label records |
| LogHub OpenStack | READY | 2,000 log records |

See `docs/data_pipeline.md`, `docs/dataset_adapters.md`, `docs/data_quality.md`, and
`docs/data_splits.md` for the Phase 2 contract and limitations.

## Storage Policy

Large raw datasets must stay out of Git. The repository tracks only import scripts, configuration, documentation, placeholders, and tests.

Ignored data folders:

- `data/raw`
- `data/external`
- `data/interim`
- `data/processed`
- `data/synthetic`
- `data/manifests`
