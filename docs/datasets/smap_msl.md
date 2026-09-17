# SMAP/MSL

## Purpose in AegisAI

SMAP/MSL provides spacecraft telemetry anomaly sequences for multivariate time-series anomaly detection.

## Source

- Official Telemanom repository: https://github.com/khundman/telemanom
- Historical data archive URL: https://s3-us-west-2.amazonaws.com/telemanom/data.zip
- Labels URL: https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
- Local destination: `data/raw/smap_msl`

## Access Method

The downloader retrieves `labeled_anomalies.csv` from GitHub and attempts to retrieve the historical `data.zip` archive. If that S3 URL returns an access error, the dataset remains incomplete and validation reports it honestly.

## File Structure

Expected raw files:

- `data.zip`
- `labeled_anomalies.csv`

The ZIP is safely extracted when available.

## Labels

Anomaly labels are in `labeled_anomalies.csv`.

## Expected Variables

To verify after archive download. Telemanom documentation describes anonymized channel telemetry and anomaly sequences.

## Approximate Size

To verify at runtime because the archive server response may change.

## License

To verify from Telemanom and the original data providers.

## AegisAI Usage

Use for multivariate anomaly detection and forecasting research after validation.

## Known Limitations

Telemetry is anonymized and not directly tied to production software service dependencies.

## Citation

Cite the Telemanom repository and the spacecraft anomaly detection paper referenced by the project.
