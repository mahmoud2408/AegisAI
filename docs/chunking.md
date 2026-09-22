# Chunking

Phase 11 uses structure-aware chunking rather than arbitrary fixed character splits.

## Strategy

Markdown documents are split by headings:

```text
document -> section -> paragraphs -> chunks
```

Plain text, HTML, and PDF text are treated as a `Document` section unless stronger structure is available.

The chunker prefers paragraph boundaries. If a paragraph is larger than the configured token window, it falls back to overlapping token windows.

## Configuration

Default config in `configs/rag/rag.yaml`:

| Setting | Value |
| --- | ---: |
| `chunk_size_tokens` | 120 |
| `chunk_overlap_tokens` | 24 |

Current measured output:

- documents: 9
- chunks: 28
- min chunk tokens: recorded in `chunk_statistics.json`
- max chunk tokens: recorded in `chunk_statistics.json`

## Chunk Provenance

Every `DocumentChunk` contains:

- `chunk_id`
- `document_id`
- `section`
- `source`
- `source_url`
- `domain`
- `version`
- `position`
- `metadata`

Retrieved chunks can therefore be cited back to their source document and section.
