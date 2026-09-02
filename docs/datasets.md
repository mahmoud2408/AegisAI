# Datasets

Phase status: provenance and import plan only.

## Primary Public Sources

| Dataset | Planned use | Source |
| --- | --- | --- |
| Numenta Anomaly Benchmark (NAB) | Univariate streaming anomaly detection and NAB scoring | https://github.com/numenta/NAB |
| Server Machine Dataset (SMD) | Multivariate server telemetry anomaly detection | https://github.com/NetManAIOps/OmniAnomaly |
| LogHub | Log parsing, event templates, log anomaly/context signals | https://github.com/logpai/loghub |

These URLs were checked during Phase 1 on 2026-09-02. Dataset import scripts will store exact commit hashes, checksums, and license/provenance metadata when implemented.

## Storage Policy

Large raw datasets must stay out of Git. The repository tracks only:

- Import scripts
- Provenance manifests
- Small schema fixtures
- Small test fixtures
- Documentation

The ignored data folders are:

- `data/raw`
- `data/external`
- `data/interim`
- `data/processed`

## Synthetic Telemetry Plan

The synthetic generator will create explicitly labeled demo or benchmark data for controlled scenarios:

- CPU overload
- Memory leak
- Traffic spike
- Network saturation
- Latency degradation
- Database bottleneck
- Service failure
- Cascading failure

Synthetic data will never be mixed with public benchmark data unless the experiment manifest records that choice.

## Provenance Manifest

Each imported dataset should produce a machine-readable manifest with:

- Dataset name
- Dataset source URL
- Source commit or version
- Download timestamp
- Checksum
- License and citation notes
- Local file paths
- Transform steps
- Split policy
- Known limitations
