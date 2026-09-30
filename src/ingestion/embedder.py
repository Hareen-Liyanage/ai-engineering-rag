"""
src/ingestion/embedder.py — Embedding generation with batching and validation

Provides:
  - Embedder interface supporting Ollama (default) and OpenAI
  - Batching to prevent timeouts on large documents
  - Dimension check matching settings.embedding_dim
"""
from __future__ import annotations

import logging
from typing import Sequence

from src.config import settings

logger = logging.getLogger(__name__)

_embedder_instance: Embedder | None = None


class Embedder:
    """Wrapper around vector embedding models."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        expected_dim: int | None = None,
    ) -> None:
        self.provider = provider or settings.llm_provider
        self.expected_dim = expected_dim or settings.embedding_dim

        if self.provider == "openai" and settings.openai_api_key:
            from langchain_openai import OpenAIEmbeddings

            self.model_name = model_name or "text-embedding-3-small"
            self._client = OpenAIEmbeddings(
                model=self.model_name,
                api_key=settings.openai_api_key,
            )
            logger.info("Embedder initialized with OpenAI (%s)", self.model_name)
        else:
            # Default to Ollama nomic-embed-text
            from langchain_ollama import OllamaEmbeddings

            self.provider = "ollama"
            self.model_name = model_name or settings.ollama_embed_model
            self._client = OllamaEmbeddings(
                base_url=settings.ollama_base_url,
                model=self.model_name,
            )
            logger.info("Embedder initialized with Ollama (%s)", self.model_name)

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of text strings in batches."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = list(texts[i : i + batch_size])
            batch_embeddings = self._client.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        # Validate dimension on first vector
        if all_embeddings and len(all_embeddings[0]) != self.expected_dim:
            logger.warning(
                "Embedding dimension mismatch: got %d, expected %d. "
                "Check EMBEDDING_DIM in your .env",
                len(all_embeddings[0]),
                self.expected_dim,
            )

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval."""
        embedding = self._client.embed_query(query)
        if len(embedding) != self.expected_dim:
            logger.warning(
                "Query embedding dimension mismatch: got %d, expected %d",
                len(embedding),
                self.expected_dim,
            )
        return embedding


def get_embedder() -> Embedder:
    """Singleton getter for the default embedder."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
