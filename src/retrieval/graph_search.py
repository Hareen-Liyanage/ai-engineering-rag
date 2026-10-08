"""Entity-linked graph retrieval with parameterized Cypher templates."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.db.neo4j_conn import get_driver

logger = logging.getLogger(__name__)

_ENTITY_PROMPT = """Extract the named entities explicitly mentioned in the question.
Return only JSON in this format: {\"entities\": [\"name\", ...]}.
Do not include generic words such as who, what, where, or why."""

GRAPH_QUERY_TEMPLATES = {
    "manages": """
        MATCH (s:Entity)-[r]->(t:Entity)
        WHERE type(r) = 'MANAGES'
          AND (elementId(s) IN $node_ids OR elementId(t) IN $node_ids)
          AND ($doc_ids IS NULL OR r.doc_id IN $doc_ids)
        RETURN s.canonical_name AS source, s.entity_type AS source_type,
               type(r) AS relation, t.canonical_name AS target,
               t.entity_type AS target_type, r.source_chunk_id AS source_chunk_id,
               r.doc_id AS doc_id
        ORDER BY r.confidence DESC
        LIMIT $limit
    """,
    "relationships": """
        MATCH (s:Entity)-[r]->(t:Entity)
        WHERE type(r) <> 'MENTIONS'
          AND (elementId(s) IN $node_ids OR elementId(t) IN $node_ids)
          AND ($doc_ids IS NULL OR r.doc_id IN $doc_ids)
        RETURN s.canonical_name AS source, s.entity_type AS source_type,
               type(r) AS relation, t.canonical_name AS target,
               t.entity_type AS target_type, r.source_chunk_id AS source_chunk_id,
               r.doc_id AS doc_id
        ORDER BY r.confidence DESC
        LIMIT $limit
    """,
    "multi_hop": """
        MATCH path = (start:Entity)-[*1..$max_hops]->(end:Entity)
        WHERE (elementId(start) IN $node_ids OR elementId(end) IN $node_ids)
          AND all(type(rel) <> 'MENTIONS' FOR rel IN relationships(path))
          AND ($doc_ids IS NULL OR all(rel.doc_id IN $doc_ids FOR rel IN relationships(path)))
        UNWIND range(0, length(nodes(path)) - 2) AS index
        WITH path, index, nodes(path)[index] AS source, relationships(path)[index] AS rel,
             nodes(path)[index + 1] AS target
        RETURN source.canonical_name AS source, source.entity_type AS source_type,
               type(rel) AS relation, target.canonical_name AS target,
               target.entity_type AS target_type, rel.source_chunk_id AS source_chunk_id,
               rel.doc_id AS doc_id
        ORDER BY rel.confidence DESC
        LIMIT $limit
    """,
}


def extract_question_entities(question: str, llm: Any | None = None) -> list[str]:
    """Use the configured chat model to identify names to resolve in Neo4j."""
    if llm is None:
        from src.graph.extractor import _get_llm

        llm = _get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=_ENTITY_PROMPT),
            HumanMessage(content=question),
        ]
    )
    raw = str(response.content).strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fenced:
        raw = fenced.group(1).strip()
    parsed = json.loads(raw)
    names = parsed.get("entities", [])
    if not isinstance(names, list):
        return []
    return list(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))


def search(
    question: str,
    doc_ids: list[int] | None = None,
    top_k: int | None = None,
    llm: Any | None = None,
    driver: Any | None = None,
    max_hops: int = 3,
) -> list[dict]:
    """Extract entities, resolve Neo4j element IDs, and query bounded graph paths."""
    entity_names = extract_question_entities(question, llm=llm)
    if not entity_names:
        return []

    from src.db.neo4j_conn import get_driver

    driver = driver or get_driver()
    if driver is None:
        return []

    try:
        with driver.session() as session:
            resolved = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.canonical_name) IN $names
                   OR ANY(alias IN coalesce(e.aliases, []) WHERE toLower(alias) IN $names)
                RETURN elementId(e) AS node_id
                """,
                {"names": [name.lower() for name in entity_names]},
            )
            node_ids = [record["node_id"] for record in resolved]
            if not node_ids:
                return []

            template_name = "manages" if re.search(r"\bmanage[sd]?\b", question, re.IGNORECASE) else "multi_hop"
            records = session.run(
                GRAPH_QUERY_TEMPLATES[template_name],
                {
                    "node_ids": node_ids,
                    "doc_ids": doc_ids or None,
                    "limit": top_k or 10,
                    "max_hops": max(1, min(3, max_hops)),
                },
            )
            return [dict(record) for record in records]
    except Exception:
        logger.exception("Graph retrieval failed for question entity names: %s", entity_names)
        return []


def triples_to_prose(triples: list[dict]) -> list[dict]:
    """Render graph triples as concise, source-attributed evidence records."""
    prose = []
    seen: set[tuple[str, str, str, int | None]] = set()
    for triple in triples:
        key = (
            str(triple.get("source", "")).casefold(),
            str(triple.get("relation", "")).casefold(),
            str(triple.get("target", "")).casefold(),
            triple.get("source_chunk_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        sentence = (
            f"{triple.get('source')} ({triple.get('source_type')}) "
            f"{str(triple.get('relation', '')).replace('_', ' ').lower()} "
            f"{triple.get('target')} ({triple.get('target_type')})."
        )
        prose.append({**triple, "text": sentence, "source": "graph"})
    return prose