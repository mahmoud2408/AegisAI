# NAB

## Purpose in AegisAI

NAB is the first univariate anomaly-detection benchmark for AegisAI. It will support statistical and Isolation Forest baselines before deeper models are added.

## Source

- Official URL: https://github.com/numenta/NAB
- Access method: public GitHub repository files
- Local destination: `data/raw/nab`

## File Structure

The downloader preserves:

- `data/`
- `labels/`
- upstream `README.md`
- upstream `LICENSE.txt`

Expected validation files:

- `data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv`
- `labels/combined_windows.json`

## Labels

NAB includes anomaly windows under `labels/`. These labels must remain untouched in raw storage.

## Expected Variables

NAB time-series CSVs generally contain timestamps and metric values. Exact schemas should be validated per file in the anomaly-detection phase.

## Approximate Size

Small, typically tens of MiB for the selected data and label files.

## License

MIT license in the upstream repository.

## AegisAI Usage

Use for streaming anomaly detection, detection delay, and NAB scoring.

## Known Limitations

Mostly univariate. It is useful for detection timing but not enough for full incident diagnosis.

## Citation

Use the citation guidance from the upstream NAB repository and related NAB publications.
