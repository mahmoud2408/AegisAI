"""Structure-aware document chunking for Phase 11 RAG."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from aegis_ai.rag.models import DocumentChunk, KnowledgeDocument, Metadata

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+")


@dataclass(frozen=True)
class ChunkingConfig:
    """Configurable chunking policy."""

    chunk_size_tokens: int = 160
    chunk_overlap_tokens: int = 30
    separators: tuple[str, ...] = ("\n\n", "\n", ". ")

    def __post_init__(self) -> None:
        if self.chunk_size_tokens < 20:
            raise ValueError("chunk_size_tokens must be at least 20")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be non-negative")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")


class StructureAwareChunker:
    """Chunk documents by section and paragraph before using word windows."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk_document(self, document: KnowledgeDocument) -> list[DocumentChunk]:
        """Create traceable chunks for one document."""

        chunks: list[DocumentChunk] = []
        position = 0
        for section, text in split_sections(document.content, document.document_type):
            for chunk_text in self._chunk_section(text):
                metadata: Metadata = dict(document.metadata)
                metadata.update(
                    {
                        "document_title": document.title,
                        "document_type": document.document_type,
                        "language": document.language,
                        "chunk_size_tokens": self.config.chunk_size_tokens,
                        "chunk_overlap_tokens": self.config.chunk_overlap_tokens,
                    }
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=stable_chunk_id(
                            document.document_id, section, position, chunk_text
                        ),
                        document_id=document.document_id,
                        text=chunk_text,
                        section=section,
                        source=document.source,
                        source_url=document.source_url,
                        domain=document.domain,
                        version=document.version,
                        position=position,
                        metadata=metadata,
                    )
                )
                position += 1
        return chunks

    def chunk_documents(self, documents: list[KnowledgeDocument]) -> list[DocumentChunk]:
        """Create chunks for many documents."""

        chunks: list[DocumentChunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def _chunk_section(self, text: str) -> list[str]:
        paragraphs = split_paragraphs(text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for paragraph in paragraphs:
            tokens = tokenize(paragraph)
            if not tokens:
                continue
            if len(tokens) > self.config.chunk_size_tokens:
                if current:
                    chunks.append("\n\n".join(current).strip())
                    current = []
                    current_tokens = 0
                chunks.extend(word_window_chunks(tokens, self.config))
                continue
            if current and current_tokens + len(tokens) > self.config.chunk_size_tokens:
                chunks.append("\n\n".join(current).strip())
                overlap = overlap_tail(current, self.config.chunk_overlap_tokens)
                current = [overlap] if overlap else []
                current_tokens = len(tokenize(overlap)) if overlap else 0
            current.append(paragraph)
            current_tokens += len(tokens)
        if current:
            chunks.append("\n\n".join(current).strip())
        return [chunk for chunk in chunks if chunk.strip()]


def split_sections(content: str, document_type: str) -> list[tuple[str, str]]:
    """Split text into named sections, preserving Markdown headings where available."""

    if document_type == "markdown":
        sections: list[tuple[str, list[str]]] = []
        current_section = "Document"
        current_lines: list[str] = []
        for line in content.splitlines():
            heading = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if heading:
                if current_lines:
                    sections.append((current_section, current_lines))
                current_section = heading.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            sections.append((current_section, current_lines))
        return [
            (section, "\n".join(lines).strip())
            for section, lines in sections
            if "\n".join(lines).strip()
        ]
    return [("Document", content.strip())] if content.strip() else []


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraph-like units."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if paragraphs:
        return paragraphs
    return [text.strip()] if text.strip() else []


def tokenize(text: str) -> list[str]:
    """Return simple lexical tokens used for chunk sizing and retrieval."""

    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def word_window_chunks(tokens: list[str], config: ChunkingConfig) -> list[str]:
    """Split a long paragraph with overlap."""

    chunks: list[str] = []
    step = config.chunk_size_tokens - config.chunk_overlap_tokens
    start = 0
    while start < len(tokens):
        window = tokens[start : start + config.chunk_size_tokens]
        chunks.append(" ".join(window))
        if start + config.chunk_size_tokens >= len(tokens):
            break
        start += step
    return chunks


def overlap_tail(paragraphs: list[str], overlap_tokens: int) -> str:
    """Return the last overlap_tokens from accumulated paragraphs."""

    if overlap_tokens == 0:
        return ""
    tokens = tokenize(" ".join(paragraphs))
    return " ".join(tokens[-overlap_tokens:])


def stable_chunk_id(
    document_id: str,
    section: str,
    position: int,
    text: str,
) -> str:
    """Create a stable chunk id."""

    digest = hashlib.sha1(f"{document_id}|{section}|{position}|{text}".encode()).hexdigest()
    return f"{document_id}_chunk_{position:04d}_{digest[:8]}"
