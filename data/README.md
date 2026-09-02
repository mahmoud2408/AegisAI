# Data Directory

Raw public datasets and generated telemetry must not be committed to Git.

Expected layout:

- `raw/`: immutable source downloads, excluded from Git except `.gitkeep`
- `external/`: externally prepared data, excluded from Git except `.gitkeep`
- `interim/`: validated and partially transformed data, excluded from Git except `.gitkeep`
- `processed/`: feature matrices, labels, and benchmark-ready splits, excluded from Git except `.gitkeep`

Every dataset import script must write provenance metadata that records:

- Dataset name and version or commit hash when available
- Source URL
- Download timestamp
- License or usage terms
- Checksum
- Transformations applied
- Train/validation/test split policy

Synthetic telemetry must be explicitly labeled as synthetic/demo data.
