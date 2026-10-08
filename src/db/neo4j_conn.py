"""
src/db/neo4j_conn.py — Neo4j connection and schema/constraint management

Responsibilities:
  - Driver singleton (bolt connection)
  - Schema init: uniqueness constraints and indexes
  - Low-level write helpers used by the graph pipeline
  - Connectivity check with graceful degradation

The driver is lazy — it only connects when first used, so the app starts
even if Neo4j Desktop is not running. Missing Neo4j disables graph
retrieval but leaves vector RAG fully functional.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - exercised only when the optional driver is absent
    GraphDatabase = None  # type: ignore[assignment]

from src.config import settings

logger = logging.getLogger(__name__)

_driver: Any | None = None


def get_driver() -> Any | None:
    """
    Return the Neo4j driver singleton, or None if the server is unreachable.
    Callers must handle None gracefully (graph path disabled).
    """
    global _driver
    if _driver is not None:
        return _driver
    if GraphDatabase is None:
        logger.warning("Neo4j Python driver is not installed; graph retrieval is disabled.")
        return None
    auth = (settings.neo4j_user, settings.neo4j_password) if settings.neo4j_password else None
    for attempt_auth in [auth, None]:
        try:
            drv = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=attempt_auth,
                connection_timeout=5,
            )
            drv.verify_connectivity()
            _driver = drv
            logger.info("Neo4j connected at %s", settings.neo4j_uri)
            return _driver
        except Exception as exc:
            logger.debug("Failed Neo4j connection attempt with auth=%s: %s", bool(attempt_auth), exc)
            continue

    logger.warning(
        "Neo4j not reachable at %s — graph retrieval disabled. "
        "Open Neo4j Desktop and start your DBMS to enable it.",
        settings.neo4j_uri,
    )
    _driver = None
    return _driver


def is_neo4j_available() -> bool:
    return get_driver() is not None


@contextmanager
def get_session() -> Generator:
    """Yield a Neo4j session; skips if driver is unavailable."""
    driver = get_driver()
    if driver is None:
        raise RuntimeError("Neo4j is not available.")
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


# ── Schema init ───────────────────────────────────────────────────────────────

_CONSTRAINTS = [
    # Entity nodes are uniquely identified by (type, canonical_name)
    "CREATE CONSTRAINT entity_unique IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.entity_type, e.canonical_name) IS UNIQUE",

    # Documents are uniquely identified by sha256
    "CREATE CONSTRAINT document_sha256 IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.sha256 IS UNIQUE",
]

_INDEXES = [
    # Full-text search on entity names (for entity resolution)
    "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name)",
]


def init_schema() -> None:
    """Create constraints and indexes. Safe to call on every startup."""
    driver = get_driver()
    if driver is None:
        logger.warning("Skipping Neo4j schema init — server not available.")
        return
    with driver.session() as session:
        for stmt in _CONSTRAINTS + _INDEXES:
            session.run(stmt)
    logger.info("Neo4j schema initialised.")


# ── Smoke-test helper ─────────────────────────────────────────────────────────

def smoke_test() -> dict:
    """
    Create a test node, read it back, delete it.
    Returns {'ok': True} or {'ok': False, 'error': str}.
    """
    driver = get_driver()
    if driver is None:
        return {"ok": False, "error": "Neo4j not reachable"}
    try:
        with driver.session() as session:
            result = session.run(
                "CREATE (t:_SmokeTest {name: 'smoke'}) RETURN t.name AS name"
            )
            name = result.single()["name"]
            session.run("MATCH (t:_SmokeTest) DELETE t")
        return {"ok": True, "node_name": name}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
