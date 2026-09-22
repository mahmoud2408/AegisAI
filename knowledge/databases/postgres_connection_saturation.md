---
document_id: postgres_connection_saturation
title: PostgreSQL Connection Saturation Runbook
source: project-generated
source_url:
version: "2026-09-22"
domain: databases
technology: postgresql
language: en
provenance: project-generated synthetic operational note
---

# PostgreSQL Connection Saturation Runbook

## Scope

This project-generated runbook supports retrieval experiments for database bottleneck incidents. It is not official PostgreSQL documentation.

## Connection saturation

When PostgreSQL reaches connection saturation, check active connection count, max_connections, connection pool size, idle-in-transaction sessions, long-running queries, lock waits, and application retry behavior.

High application latency can appear when all pool workers wait for database connections. Database response time, query duration, and lock wait metrics should be correlated with service error rate.

## Evidence

Capture query fingerprints, wait events, lock graph, connection pool configuration, slow query samples, and timestamps for saturation onset.
