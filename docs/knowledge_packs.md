# Knowledge Packs

Phase 11 introduces reusable knowledge packs as local directories under `knowledge/`.

## Current Packs

| Pack | Domain | Documents | Source type |
| --- | --- | ---: | --- |
| `openstack` | `cloud_infrastructure` | 1 | project-generated |
| `linux` | `operating_system` | 1 | project-generated |
| `docker` | `containers` | 1 | project-generated |
| `kubernetes` | `containers` | 1 | project-generated |
| `databases` | `databases` | 1 | project-generated |
| `networking` | `networking` | 1 | project-generated |
| `monitoring` | `monitoring` | 1 | project-generated |
| `incident_runbooks` | `incident_response` | 1 | project-generated |
| `synthetic` | `industrial` | 1 | synthetic |

## Pack Contents

A production knowledge pack should contain:

- source documents;
- document metadata;
- chunk artifacts;
- embedding/index metadata;
- evaluation queries;
- version information.

The current implementation stores shared pack artifacts in:

```text
experiments/rag/phase11_rag_knowledge_20260922
```

## Provenance Policy

Do not present project-generated notes as official documentation. Official, third-party, project-generated, and synthetic sources must remain distinguishable through metadata.

## Future Work

Future packs should add official source URLs and versioned downloaded documentation where license and size constraints permit. Those additions should be evaluated separately rather than mixed silently into the current synthetic/project-generated pack.
