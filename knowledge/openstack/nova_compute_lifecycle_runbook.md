---
document_id: openstack_nova_compute_lifecycle
title: OpenStack Nova Compute Lifecycle Triage Runbook
source: project-generated
source_url:
version: "2026-09-22"
domain: cloud_infrastructure
technology: openstack
language: en
provenance: project-generated synthetic operational note, not official OpenStack documentation
---

# OpenStack Nova Compute Lifecycle Triage Runbook

## Scope

This project-generated runbook supports AegisAI retrieval experiments for Nova compute lifecycle issues. It is not official OpenStack documentation.

## Repeated VM lifecycle errors

When nova-compute reports repeated VM lifecycle problems, first identify the instance UUID, request ID, host, and lifecycle action. Compare the log event template against nearby libvirt, scheduler, and conductor messages.

Useful checks:

- confirm nova-compute service health on the affected host;
- inspect libvirt domain state and recent hypervisor errors;
- check whether the scheduler or conductor retried the same request;
- compare error timestamps with host CPU, memory, disk, and network pressure;
- verify that the instance is not already in a transitional state such as pausing, unpausing, resizing, or deleting.

## Evidence to capture

Capture nova-compute component, log level, event template, request ID, instance UUID, host name, and the surrounding five minutes of logs. Treat repeated lifecycle messages as investigation evidence, not proof of root cause.

## Remediation guardrails

Avoid restarting nova-compute until active migrations and rebuilds have been checked. Prefer read-only service and libvirt inspection before remediation.
