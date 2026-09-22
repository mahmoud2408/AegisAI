# Document Ingestion

Phase 11 loads only documents that exist locally under `knowledge/`.

## Knowledge Layout

```text
knowledge/
├── openstack/
├── linux/
├── docker/
├── kubernetes/
├── databases/
├── networking/
├── monitoring/
├── incident_runbooks/
└── synthetic/
```

## Loader Interface

All loaders implement:

- `load()`
- `extract_text()`
- `extract_metadata()`

Supported file types:

- Markdown
- plain text
- HTML
- PDF where `pypdf` is installed

HTML loading extracts visible text and ignores `script`, `style`, and `noscript` content. Documents are treated as untrusted input and are never executed.

## Provenance

Every `KnowledgeDocument` preserves:

- `document_id`
- `title`
- `source`
- `source_url`
- `version`
- `domain`
- `document_type`
- `language`
- `created_at`
- `metadata`

The current corpus contains 9 documents:

| Domain | Documents |
| --- | ---: |
| `cloud_infrastructure` | 1 |
| `containers` | 2 |
| `databases` | 1 |
| `incident_response` | 1 |
| `industrial` | 1 |
| `monitoring` | 1 |
| `networking` | 1 |
| `operating_system` | 1 |

Sources:

- `project-generated`: 8 documents
- `synthetic`: 1 document

No current Phase 11 document is represented as official vendor documentation.
