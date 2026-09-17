# SMD

## Purpose in AegisAI

SMD is the main multivariate server telemetry benchmark for anomaly detection.

## Source

- Official/public repository requested for this project: https://github.com/das2ar/OmniAnomaly
- Access method: public GitHub `ServerMachineDataset` directory
- Local destination: `data/raw/smd`

## File Structure

The downloader preserves:

- `train/`
- `test/`
- `test_label/`
- `interpretation_label/`
- `LICENSE`

## Labels

The `test_label` directory contains anomaly labels. `interpretation_label` contains additional label annotations.

## Expected Variables

Each machine file is a multivariate time-series text file. Exact dimensions and parsing rules will be validated in the SMD phase.

## Approximate Size

Hundreds of MiB when all train/test/test_label files are present.

## License

To verify from the upstream repository and original SMD publication.

## AegisAI Usage

Use for multivariate anomaly detection, temporal correlation, and early root-cause evidence experiments.

## Known Limitations

Service topology and operational context are limited, so causal conclusions must be conservative.

## Citation

Cite OmniAnomaly/SMD sources according to the upstream repository and paper guidance.
