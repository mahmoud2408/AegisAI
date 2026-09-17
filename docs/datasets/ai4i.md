# AI4I 2020

## Purpose in AegisAI

AI4I 2020 supports supervised failure prediction, calibration, and explainability workflows.

## Source

- Official URL: https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset
- Access method: UCI public ZIP archive
- Local destination: `data/raw/ai4i`

## File Structure

Expected files:

- `ai4i2020.csv`
- original ZIP archive retained locally

## Labels

The dataset includes `Machine failure` and failure-mode target columns such as TWF, HDF, PWF, OSF, and RNF.

## Expected Variables

UCI lists product/type identifiers and process variables including air temperature, process temperature, rotational speed, torque, and tool wear.

## Approximate Size

UCI lists `ai4i2020.csv` at about 510 KiB with 10,000 rows.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).

## AegisAI Usage

Use for incident/failure prediction baselines and SHAP-style explainability experiments.

## Known Limitations

The dataset is synthetic. It is useful for controlled modeling practice but should not be presented as real production telemetry.

## Citation

Cite the UCI dataset DOI and the AI4I paper listed on the UCI page.
