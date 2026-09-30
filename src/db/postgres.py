"""
src/db/postgres.py — PostgreSQL / pgvector connection and schema management

Responsibilities:
  - Connection pooling via psycopg2
  - One-time schema init (chunks table + HNSW index)
  - UPSERT helpers used by the ingestion pipeline
  - Vector search used by the retrieval layer

Schema:
  documents(id, filename, sha256, ingested_at)
  chunks(id, doc_id, chunk_index, page_number, text, embedding vector(768),
         entity_ids text[], metadata jsonb)
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Generator, Sequence

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from src.config import settings

logger = logging.getLogger(__name__)

# ── DDL ──────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT        NOT NULL,
    sha256      CHAR(64)    NOT NULL UNIQUE,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    doc_id      INTEGER     NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER     NOT NULL,
    page_number INTEGER     NOT NULL,
    text        TEXT        NOT NULL,
    embedding   vector({settings.embedding_dim}),
    entity_ids  TEXT[]      NOT NULL DEFAULT '{{}}',
    metadata    JSONB       NOT NULL DEFAULT '{{}}'
);

-- HNSW index for fast approximate nearest-neighbour search
-- ef_construction=128, m=16 are good starting defaults for this dimension
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- For exact lookups by document
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);
"""


# ── Connection ────────────────────────────────────────────────────────────────

def _make_connection() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=settings.pg_database,
    )
    # register_vector must be called outside a transaction (i.e. in autocommit mode)
    conn.autocommit = True
    register_vector(conn)
    conn.autocommit = False
    return conn


@contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager that yields a connection and commits / rolls back."""
    conn = _make_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema init ───────────────────────────────────────────────────────────────

def init_schema() -> None:
    """Create tables and indexes if they don't exist. Safe to call on every startup."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
    logger.info("PostgreSQL schema initialised.")


# ── Document helpers ──────────────────────────────────────────────────────────

def document_exists(sha256: str) -> bool:
    """Return True if a document with this hash has already been ingested."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE sha256 = %s", (sha256,))
            return cur.fetchone() is not None


def upsert_document(filename: str, sha256: str) -> int:
    """Insert a document row and return its id (or return existing id)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, sha256)
                VALUES (%s, %s)
                ON CONFLICT (sha256) DO UPDATE SET filename = EXCLUDED.filename
                RETURNING id
                """,
                (filename, sha256),
            )
            row = cur.fetchone()
            assert row is not None
            return row[0]


# ── Chunk helpers ─────────────────────────────────────────────────────────────

def upsert_chunks(
    conn: psycopg2.extensions.connection,
    doc_id: int,
    chunks: list[dict],
) -> list[int]:
    """
    Bulk-insert chunks for a document. Each dict must have:
        chunk_index, page_number, text, embedding (list[float]),
        entity_ids (list[str]), metadata (dict)

    Returns the list of inserted chunk ids.
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks (doc_id, chunk_index, page_number, text, embedding, entity_ids, metadata)
            VALUES %s
            RETURNING id
            """,
            [
                (
                    doc_id,
                    c["chunk_index"],
                    c["page_number"],
                    c["text"],
                    c["embedding"],
                    c.get("entity_ids", []),
                    json.dumps(c.get("metadata", {})),
                )
                for c in chunks
            ],
            template="(%s, %s, %s, %s, %s::vector, %s, %s::jsonb)",
        )
        return [row[0] for row in cur.fetchall()]


# ── Vector search ─────────────────────────────────────────────────────────────

def vector_search(
    query_embedding: Sequence[float],
    top_k: int = 5,
    doc_ids: list[int] | None = None,
) -> list[dict]:
    """
    Return the top_k most similar chunks by cosine distance.
    Optionally filter to a subset of document ids.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if doc_ids:
                cur.execute(
                    """
                    SELECT c.id, c.doc_id, c.chunk_index, c.page_number, c.text,
                           c.entity_ids, c.metadata,
                           1 - (c.embedding <=> %s::vector) AS score
                    FROM chunks c
                    WHERE c.doc_id = ANY(%s)
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (list(query_embedding), doc_ids, list(query_embedding), top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT c.id, c.doc_id, c.chunk_index, c.page_number, c.text,
                           c.entity_ids, c.metadata,
                           1 - (c.embedding <=> %s::vector) AS score
                    FROM chunks c
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (list(query_embedding), list(query_embedding), top_k),
                )
            return [dict(row) for row in cur.fetchall()]


def fetch_chunk_by_id(chunk_id: int) -> dict | None:
    """Fetch a single chunk by primary key — used by citation validator."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM chunks WHERE id = %s", (chunk_id,))
            row = cur.fetchone()
            return dict(row) if row else None
