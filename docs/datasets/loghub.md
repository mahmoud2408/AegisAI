# LogHub

## Purpose in AegisAI

LogHub supplies log data for parsing, event-template extraction, error-rate signals, and log evidence in incident investigations.

## Source

- Official URL: https://github.com/logpai/loghub
- Access method: public GitHub repository samples by default
- Local destinations:
  - `data/raw/loghub/hdfs`
  - `data/raw/loghub/bgl`
  - `data/raw/loghub/openstack`
  - `data/raw/loghub/hadoop`
  - `data/raw/loghub/spark`
  - `data/raw/loghub/zookeeper`

## File Structure

The safe default downloads the small 2k sample files and upstream README for each selected dataset. It does not download large full releases.

## Labels

Some LogHub datasets have structured/template files and some full datasets have labels. The safe sample set includes structured and template CSVs when provided in the upstream repository.

## Expected Variables

Raw `.log` files contain unstructured log lines. Structured CSVs and template CSVs are source-provided parser outputs.

## Approximate Size

The default sample set is a few MiB. Full releases can range from MiB to tens of GiB; Thunderbird, Windows, Spark full, HDFS v2, and similar large releases are not automatic defaults.

## License

The upstream repository states the datasets are freely available for research or academic work. Cite LogHub where applicable.

## AegisAI Usage

Use for log parsing, anomaly context, template frequencies, error-rate features, and report evidence.

## Known Limitations

Small samples are useful for pipeline development but not enough for serious production-scale log anomaly benchmarks.

## Citation

Cite the LogHub papers referenced in the upstream repository.
