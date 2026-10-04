"""
src/config.py — Central configuration loaded from .env

All database URLs, API keys, and tuneable constants live here.
Import `settings` anywhere in the codebase; never read env vars directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root


@dataclass(frozen=True)
class Settings:
    # ── PostgreSQL / pgvector ────────────────────────────────────────────────
    pg_host: str = field(default_factory=lambda: os.getenv("PG_HOST", "localhost"))
    pg_port: int = field(default_factory=lambda: int(os.getenv("PG_PORT", "5433")))
    pg_user: str = field(default_factory=lambda: os.getenv("PG_USER", "rag_user"))
    pg_password: str = field(default_factory=lambda: os.getenv("PG_PASSWORD", "rag_password"))
    pg_database: str = field(default_factory=lambda: os.getenv("PG_DATABASE", "rag_db"))

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))

    # ── LLM providers ───────────────────────────────────────────────────────
    # "ollama" | "anthropic" | "openai"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_fallback_enabled: bool = field(
        default_factory=lambda: os.getenv("LLM_FALLBACK_ENABLED", "false").lower() in {"1", "true", "yes"}
    )
    llm_fallback_provider: str = field(
        default_factory=lambda: os.getenv("LLM_FALLBACK_PROVIDER", "anthropic").lower()
    )

    # ── Ollama (local dev) ───────────────────────────────────────────────────
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_chat_model: str = field(default_factory=lambda: os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b"))
    ollama_embed_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    # ── Cloud model names ────────────────────────────────────────────────────
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_words: int = field(default_factory=lambda: int(os.getenv("CHUNK_WORDS", "180")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "35")))

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768")))  # nomic-embed-text
    route_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("ROUTE_CONFIDENCE_THRESHOLD", "0.65"))
    )

    # ── Guardrails (Phase 6) ─────────────────────────────────────────────────
    guardrails_enabled: bool = field(
        default_factory=lambda: os.getenv("GUARDRAILS_ENABLED", "true").lower() in {"1", "true", "yes"}
    )
    # Minimum fraction of claim tokens that must appear in the cited chunk text
    # for a sentence to be considered grounded.  Range: 0.0 – 1.0.
    groundedness_min_overlap: float = field(
        default_factory=lambda: float(os.getenv("GROUNDEDNESS_MIN_OVERLAP", "0.25"))
    )

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


# Singleton — import this everywhere
settings = Settings()
