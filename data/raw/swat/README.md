# SWaT Manual Dataset Installation

SWaT requires requesting access from iTrust:

https://www.sutd.edu.sg/itrust/itrust-labs/datasets/

Do not bypass this access process.

After approval:

1. Download the files supplied by iTrust.
2. Preserve the provider's original filenames and folder structure.
3. Place the files under this directory: `data/raw/swat`.
4. Keep any license, terms, or documentation files supplied by iTrust.
5. Run `python scripts/datasets/validate_datasets.py`.

Validation currently detects whether this manual folder exists and reports manual access until dataset-specific expected filenames are confirmed from the approved package.
