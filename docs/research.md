# Research Plan

Phase status: research design only.

## Central Question

Does combining predictive models, telemetry analysis, retrieval-augmented generation, and agentic reasoning improve automated incident diagnosis compared with isolated machine-learning approaches?

## Ablation Study

Planned variants:

- Baseline
- ML
- ML + Forecasting
- ML + RAG
- Full Agent

## Evaluation Dimensions

- Detection performance
- Diagnosis quality
- False positives
- Latency
- Investigation success rate

## Diagnosis Quality

Diagnosis quality should be evaluated with a reproducible rubric that checks:

- Whether the affected service is identified
- Whether the evidence matches the incident window
- Whether the probable cause is supported by telemetry, logs, traces, or documentation
- Whether recommendations are actionable
- Whether the report avoids unsupported causal claims

## Artifacts

Each ablation run should save:

- Dataset manifest
- Run configuration
- Model or pipeline artifact
- Quantitative metrics
- Investigation reports
- Evaluation rubric outputs
- Reproduction command

No results should be added until the experiment scripts generate them.
