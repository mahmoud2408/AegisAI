# MetroPT-3

## Purpose in AegisAI

MetroPT-3 supports predictive maintenance, anomaly explanation, and temporal failure analysis.

## Source

- Official URL: https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset
- Access method: UCI public ZIP archive
- Local destination: `data/raw/metropt`

## File Structure

Expected files:

- `MetroPT3(AirCompressor).csv`
- `Data Description_Metro.pdf`
- original ZIP archive retained locally

## Labels

The UCI page describes failure reports rather than a simple per-row binary target. The local CSV has no machine-readable failure label column.

Phase 7 curates four air-leak failure intervals from `Data Description_Metro.pdf` and uses them to define a six-hour early-warning target. These labels are report-derived case-study labels, not native dense per-row annotations.

## Expected Variables

UCI lists pressure, temperature, motor current, and digital/electrical signal variables. Exact schema validation will be added in the MetroPT data phase.

## Approximate Size

UCI lists the data file at about 208 MiB and the description PDF at about 79 KiB.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).

## AegisAI Usage

Use for forecasting, anomaly explanation, and predictive maintenance experiments.

## Known Limitations

Failure labels require careful construction from reported event windows. Do not fabricate labels or treat the source CSV as if it contained native failure annotations.

## Citation

Cite the UCI dataset DOI and associated papers listed on the UCI page.
