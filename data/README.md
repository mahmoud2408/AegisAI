# Data Directory

Raw public datasets and generated telemetry must not be committed to Git.

## Layout

- `raw/`: immutable source downloads, excluded from Git except `.gitkeep` and README files.
- `external/`: externally prepared data, excluded from Git except `.gitkeep`.
- `interim/`: validated and partially transformed data, excluded from Git except `.gitkeep`.
- `processed/`: canonical Parquet streams and later feature matrices, excluded from Git except `.gitkeep`.
- `synthetic/`: generated telemetry scenarios, excluded from Git except `.gitkeep`.
- `manifests/`: local dataset manifests, excluded from Git except `.gitkeep`.

## Automatic Downloads

The safe public downloader currently targets:

- NAB into `data/raw/nab`
- SMD into `data/raw/smd`
- SMAP/MSL Telemanom bundle into `data/raw/smap_msl` when the public archive is reachable
- LogHub small 2k samples into `data/raw/loghub/*`
- OpenTelemetry demo reference resources into `data/raw/opentelemetry`
- MetroPT-3 into `data/raw/metropt`
- AI4I 2020 into `data/raw/ai4i`

Run:

```powershell
python scripts/datasets/download_all.py --dataset public
```

## Manual Downloads

These datasets require manual access or official download steps:

- SWaT: request access from iTrust, then place files under `data/raw/swat`.
- WADI: request access from iTrust, then place files under `data/raw/wadi`.
- CIC-IDS2017: use the official CIC download page, then place files under `data/raw/cicids2017`.

Each manual folder contains a README with dataset-specific instructions.

## Validation

Run:

```powershell
python scripts/datasets/validate_datasets.py
```

The validator checks for missing directories, missing expected files, missing labels, and corrupted ZIP archives where relevant.

## Audit

Run:

```powershell
python scripts/datasets/audit_datasets.py --pretty
```

The audit writes `data/manifests/dataset_audit.json` with measured local file counts, sizes,
schemas, row counts, label summaries, and temporal summaries. The generated JSON is ignored by
Git because it reflects the local data installation.

## Canonical Preprocessing

Run:

```powershell
python scripts/data/profile_all.py
python scripts/data/preprocess_all.py --batch-size 25000
python scripts/data/validate_processed.py
```

The preprocessing pipeline writes local generated artifacts:

- `processed/metrics/<dataset>/`
- `processed/logs/<dataset>/`
- `processed/traces/<dataset>/`
- `processed/incidents/<dataset>/`
- `processed/failures/<dataset>/`
- `processed/labels/<dataset>/`
- `manifests/profiles/<dataset>.json`
- `manifests/lineage/<dataset>.json`
- `manifests/preprocessing_summary.json`
- `manifests/preprocessing_report.md`
- `manifests/processed_validation.json`

These files are ignored by Git. Regenerate them from raw data and scripts.

## Storage Policy

Approximate storage requirements depend on what you select:

- NAB: small, tens of MiB.
- SMD: hundreds of MiB.
- SMAP/MSL: To verify at runtime because the historical public S3 URL may be unavailable.
- LogHub safe samples: small, a few MiB.
- MetroPT-3: about 208 MiB from UCI.
- AI4I: about 510 KiB from UCI.
- SWaT/WADI/CIC-IDS2017: To verify after manual download.

Do not commit raw datasets, processed datasets, synthetic generated data, model artifacts, or local manifests.
