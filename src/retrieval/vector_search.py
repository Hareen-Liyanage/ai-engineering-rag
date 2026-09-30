"""Vector retrieval adapter for the PostgreSQL/pgvector store."""
from __future__ import annotations

from src.config import settings
from src.db.postgres import vector_search as search_vectors
from src.ingestion.embedder import Embedder, get_embedder


def search(
    question: str,
    top_k: int | None = None,
    doc_ids: list[int] | None = None,
    embedder: Embedder | None = None,
) -> list[dict]:
    """Embed a question and retrieve its nearest document chunks."""
    query_embedding = (embedder or get_embedder()).embed_query(question)
    return search_vectors(
        query_embedding=query_embedding,
        top_k=top_k or settings.top_k,
        doc_ids=doc_ids,
    )