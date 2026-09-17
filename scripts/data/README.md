# Data Preprocessing CLI

Phase 2 data-layer commands:

```powershell
python scripts/data/profile_all.py
python scripts/data/profile_all.py --dataset smd
python scripts/data/preprocess_dataset.py --dataset nab
python scripts/data/preprocess_dataset.py --dataset metropt
python scripts/data/preprocess_all.py --batch-size 25000
python scripts/data/validate_processed.py
```

Generated outputs:

- `data/processed/<canonical_type>/<dataset>/.../*.parquet`
- `data/manifests/profiles/<dataset>.json`
- `data/manifests/lineage/<dataset>.json`
- `data/manifests/preprocessing_summary.json`
- `data/manifests/preprocessing_report.md`

The commands are idempotent: each run regenerates only the selected dataset's processed outputs.
Raw data is read-only. SMD and MetroPT produce large long-format metric outputs, so use a smaller
batch size when memory is constrained.
