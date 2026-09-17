# OpenTelemetry Demo Resources

## Purpose in AegisAI

OpenTelemetry demo resources will help design future telemetry-generation and observability pipelines.

## Source

- Official/public resource used here: https://github.com/openobserve/opentelemetry-demo-dataset
- Access method: selected public GitHub files
- Local destination: `data/raw/opentelemetry`

## File Structure

The downloader intentionally retrieves only small reference files such as:

- `README.md`
- `src/log-generator/app.py`
- `src/otel-collector/otelcol-config.yml`
- `src/prometheus/prometheus-config.yaml`
- selected Grafana dashboard JSON files
- selected trace-test configuration files

## Labels

No static incident labels are provided by the inspected reference repository.

## Expected Variables

Not applicable yet. This is reference material, not a static telemetry benchmark.

## Approximate Size

Small, generally less than a few MiB for the selected files.

## License

Apache License 2.0 in the upstream repository.

## AegisAI Usage

Use as reference for future synthetic telemetry generation and observability ingestion design.

## Known Limitations

The inspected repository is a runnable demo where data starts flowing when services run; it is not a curated static telemetry dataset.

## Citation

Reference the upstream repository and OpenTelemetry/OpenObserve docs where used.
