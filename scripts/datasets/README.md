# Dataset Scripts

The main entry point is:

```powershell
python scripts/datasets/download_all.py --dataset public
```

Useful commands:

```powershell
python scripts/datasets/download_all.py --list
python scripts/datasets/download_all.py --dataset nab
python scripts/datasets/download_all.py --dataset smd
python scripts/datasets/download_all.py --dataset smap-msl
python scripts/datasets/download_all.py --dataset loghub
python scripts/datasets/download_all.py --dataset hdfs
python scripts/datasets/download_all.py --dataset public
python scripts/datasets/download_all.py --dataset all --large
python scripts/datasets/validate_datasets.py
python scripts/datasets/audit_datasets.py --pretty
```

The downloader updates `data/manifests/datasets_manifest.json`, which is intentionally ignored by Git.
The audit command updates `data/manifests/dataset_audit.json`, which is also ignored by Git
because it reflects the local raw-data installation.
