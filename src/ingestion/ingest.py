"""
src/ingestion/ingest.py — Idempotent document ingestion pipeline

Flow:
  1. Hash document bytes with SHA-256.
  2. If already in PostgreSQL (and force=False), skip without doing any redundant work.
  3. Chunk document with page provenance.
  4. Generate dense vector embeddings via Embedder.
  5. Upsert document record and insert all chunks into pgvector.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Sequence

import psycopg2.extras

from src.db import postgres
from src.ingestion.chunker import TextChunk, extract_chunks_from_pdf
from src.ingestion.embedder import Embedder, get_embedder

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    doc_id: int | None
    filename: str
    sha256: str
    chunks_count: int
    status: str  # "ingested" | "skipped" | "error"
    message: str
    elapsed_seconds: float = 0.0


def compute_sha256(data: bytes) -> str:
    """Compute standard SHA-256 hexadecimal digest."""
    return hashlib.sha256(data).hexdigest()


def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    force: bool = False,
    embedder: Embedder | None = None,
    build_graph: bool = True,
) -> IngestionResult:
    """
    Ingest a PDF file into PostgreSQL/pgvector.
    Idempotent: skips processing if the document has already been ingested.
    """
    t0 = time.perf_counter()
    file_hash = compute_sha256(pdf_bytes)

    # 1. Check idempotency
    if not force and postgres.document_exists(file_hash):
        with postgres.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT d.id, COUNT(c.id)
                    FROM documents d
                    LEFT JOIN chunks c ON c.doc_id = d.id
                    WHERE d.sha256 = %s
                    GROUP BY d.id
                    """,
                    (file_hash,),
                )
                row = cur.fetchone()
                existing_doc_id = row[0] if row else None
                chunk_count = row[1] if row else 0

        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(
            "Document '%s' (hash: %s...) already ingested as doc_id=%s. Skipping.",
            filename,
            file_hash[:8],
            existing_doc_id,
        )
        return IngestionResult(
            doc_id=existing_doc_id,
            filename=filename,
            sha256=file_hash,
            chunks_count=chunk_count,
            status="skipped",
            message=f"Document already exists (doc_id={existing_doc_id}). Skipped re-ingestion.",
            elapsed_seconds=elapsed,
        )

    # 2. Extract chunks
    try:
        chunks: list[TextChunk] = extract_chunks_from_pdf(pdf_bytes, filename=filename)
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 3)
        logger.exception("Failed to parse PDF '%s': %s", filename, exc)
        return IngestionResult(
            doc_id=None,
            filename=filename,
            sha256=file_hash,
            chunks_count=0,
            status="error",
            message=f"PDF parsing error: {exc}",
            elapsed_seconds=elapsed,
        )

    if not chunks:
        elapsed = round(time.perf_counter() - t0, 3)
        return IngestionResult(
            doc_id=None,
            filename=filename,
            sha256=file_hash,
            chunks_count=0,
            status="error",
            message="No text chunks could be extracted from this PDF.",
            elapsed_seconds=elapsed,
        )

    # 3. Generate embeddings
    embedder_instance = embedder or get_embedder()
    try:
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder_instance.embed_texts(texts)
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 3)
        logger.exception("Embedding failed for '%s': %s", filename, exc)
        return IngestionResult(
            doc_id=None,
            filename=filename,
            sha256=file_hash,
            chunks_count=len(chunks),
            status="error",
            message=f"Embedding error: {exc}",
            elapsed_seconds=elapsed,
        )

    # 4. Upsert document and chunks in PostgreSQL
    with postgres.get_conn() as conn:
        with conn.cursor() as cur:
            # Upsert document
            cur.execute(
                """
                INSERT INTO documents (filename, sha256)
                VALUES (%s, %s)
                ON CONFLICT (sha256) DO UPDATE SET filename = EXCLUDED.filename
                RETURNING id
                """,
                (filename, file_hash),
            )
            doc_id = cur.fetchone()[0]

            # If force re-ingesting, clear old chunks for this document
            if force:
                cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

        chunk_payloads = [
            {
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "text": c.text,
                "embedding": emb,
                "entity_ids": [],
                "metadata": {
                    **c.metadata,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                },
            }
            for c, emb in zip(chunks, embeddings)
        ]

        chunk_ids = postgres.upsert_chunks(conn, doc_id, chunk_payloads)

    # 5. Extract entities and relationships to build Neo4j graph
    if build_graph:
        try:
            from src.graph.extractor import extract_knowledge_from_chunk
            from src.graph.writer import write_extraction_to_graph

            extractions = []
            for cid, chunk in zip(chunk_ids, chunks):
                ext = extract_knowledge_from_chunk(chunk.text, chunk_id=cid, doc_id=doc_id)
                if ext.entities or ext.relationships:
                    extractions.append(ext)

            if extractions:
                write_extraction_to_graph(extractions, filename=filename)
        except Exception as exc:
            logger.warning("Graph building encountered an error for '%s': %s", filename, exc)

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(
        "Successfully ingested '%s': %d chunks into doc_id=%s in %.2fs",
        filename,
        len(chunks),
        doc_id,
        elapsed,
    )

    return IngestionResult(
        doc_id=doc_id,
        filename=filename,
        sha256=file_hash,
        chunks_count=len(chunks),
        status="ingested",
        message=f"Successfully ingested {len(chunks)} chunks in {elapsed}s.",
        elapsed_seconds=elapsed,
    )


def list_ingested_documents() -> list[dict]:
    """Return all ingested documents with their chunk counts and ingestion dates."""
    with postgres.get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT d.id, d.filename, d.sha256, d.ingested_at,
                       COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.doc_id = d.id
                GROUP BY d.id
                ORDER BY d.ingested_at DESC
                """
            )
            return [dict(row) for row in cur.fetchall()]


def delete_document(doc_id: int) -> bool:
    """Delete a document and cascade delete its chunks and graph nodes."""
    try:
        from src.graph.writer import clear_graph_for_document

        clear_graph_for_document(doc_id)
    except Exception as exc:
        logger.warning("Could not clear graph for doc_id=%d: %s", doc_id, exc)

    with postgres.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s RETURNING id", (doc_id,))
            return cur.fetchone() is not None
