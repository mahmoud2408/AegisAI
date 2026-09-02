# Agent

Phase status: design proposal only.

## Investigation Agent

The investigation agent will use validated tools to inspect telemetry, logs, traces, historical incidents, models, forecasts, and technical documentation.

Planned tools:

- `get_metrics`
- `get_logs`
- `get_traces`
- `get_service_dependencies`
- `get_incident_history`
- `run_anomaly_detection`
- `run_forecast`
- `run_incident_prediction`
- `search_documentation`

## Guardrails

The agent must implement:

- Structured tool schemas
- Input validation
- Retries for transient failures
- Timeouts
- Structured logging
- Maximum tool-call limits
- Clear error states
- No hidden chain-of-thought disclosure

## Report Schema

Required report fields:

- Incident summary
- Severity
- Confidence
- Evidence
- Probable root causes
- Affected services
- Recommended actions
- Supporting documentation

Reports must distinguish observed evidence from model inference and recommendations.
