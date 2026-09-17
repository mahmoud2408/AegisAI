# CIC-IDS2017 Manual Dataset Installation

Use the official CIC-IDS2017 page:

https://www.unb.ca/cic/datasets/ids-2017.html

Do not invent direct download URLs and do not rely on unofficial mirrors unless explicitly approved and documented.

After downloading from the official process:

1. Place the official archives under this directory: `data/raw/cicids2017`.
2. Preserve original archive names such as `MachineLearningCSV.zip` and `GeneratedLabelledFlows.zip` when provided.
3. Keep any license, citation, or usage files.
4. Run `python scripts/datasets/validate_datasets.py`.

The official page describes labeled flows, PCAPs, profiles, and CSV files for machine learning. Current file availability and terms should be verified during manual download.
