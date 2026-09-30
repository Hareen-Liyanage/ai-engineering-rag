"""
src/ingestion/chunker.py — Rich chunking with page provenance and offsets

Produces `TextChunk` objects with:
  - doc-relative chunk index
  - page number
  - character offsets within the page
  - word count and metadata
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from io import BytesIO

from pypdf import PdfReader

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    page_number: int
    text: str
    char_start: int
    char_end: int
    section: str = ""
    metadata: dict = field(default_factory=dict)


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_chunks_from_pdf(
    pdf_bytes: bytes,
    filename: str = "",
    chunk_words: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """
    Extract text from a PDF, chunking page-by-page using a sliding word window.
    Maintains page numbers and approximate character offsets within each page.
    """
    chunk_words = chunk_words or settings.chunk_words
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    step = max(1, chunk_words - chunk_overlap)

    reader = PdfReader(BytesIO(pdf_bytes))
    chunks: list[TextChunk] = []
    chunk_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = _clean_text(raw_text)
        if not cleaned:
            continue

        words = cleaned.split(" ")
        if not words or not any(w.strip() for w in words):
            continue

        # Build character index mapping for words to determine char_start and char_end
        word_spans: list[tuple[int, int]] = []
        pos = 0
        for w in words:
            start = cleaned.find(w, pos)
            if start == -1:
                start = pos
            end = start + len(w)
            word_spans.append((start, end))
            pos = end

        for start_idx in range(0, len(words), step):
            end_idx = min(start_idx + chunk_words, len(words))
            chunk_slice = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_slice).strip()

            if not chunk_text:
                continue

            char_start = word_spans[start_idx][0] if start_idx < len(word_spans) else 0
            char_end = word_spans[end_idx - 1][1] if end_idx - 1 < len(word_spans) else len(cleaned)

            chunks.append(
                TextChunk(
                    chunk_index=chunk_index,
                    page_number=page_num,
                    text=chunk_text,
                    char_start=char_start,
                    char_end=char_end,
                    metadata={
                        "filename": filename,
                        "word_count": len(chunk_slice),
                        "total_pages": len(reader.pages),
                    },
                )
            )
            chunk_index += 1

            if end_idx == len(words):
                break

    logger.info("Extracted %d chunks from %s (%d pages)", len(chunks), filename, len(reader.pages))
    return chunks


def extract_chunks_from_text(
    text: str,
    filename: str = "text_document",
    page_number: int = 1,
    chunk_words: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """Chunk arbitrary plain text using sliding word window."""
    chunk_words = chunk_words or settings.chunk_words
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    step = max(1, chunk_words - chunk_overlap)

    cleaned = _clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split(" ")
    chunks: list[TextChunk] = []

    word_spans: list[tuple[int, int]] = []
    pos = 0
    for w in words:
        start = cleaned.find(w, pos)
        if start == -1:
            start = pos
        end = start + len(w)
        word_spans.append((start, end))
        pos = end

    for chunk_index, start_idx in enumerate(range(0, len(words), step)):
        end_idx = min(start_idx + chunk_words, len(words))
        chunk_slice = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_slice).strip()

        if not chunk_text:
            continue

        char_start = word_spans[start_idx][0] if start_idx < len(word_spans) else 0
        char_end = word_spans[end_idx - 1][1] if end_idx - 1 < len(word_spans) else len(cleaned)

        chunks.append(
            TextChunk(
                chunk_index=chunk_index,
                page_number=page_number,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                metadata={
                    "filename": filename,
                    "word_count": len(chunk_slice),
                },
            )
        )
        if end_idx == len(words):
            break

    return chunks
