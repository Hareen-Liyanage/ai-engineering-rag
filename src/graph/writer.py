"""
src/graph/writer.py — Neo4j graph construction and querying
"""
from __future__ import annotations

import logging
from typing import Sequence

from src.db.neo4j_conn import get_driver
from src.graph.ontology import (
    RELATIONSHIP_TYPES,
    ExtractionResult,
)
from src.graph.resolver import EntityResolver

logger = logging.getLogger(__name__)


def write_extraction_to_graph(
    results: Sequence[ExtractionResult],
    filename: str = "",
    resolver: EntityResolver | None = None,
) -> dict:
    """
    Write extracted entities and relationships into Neo4j in batch transactions.
    Idempotent: uses MERGE for nodes and relationships.
    """
    driver = get_driver()
    if driver is None:
        logger.warning("Neo4j not available — skipping graph writing.")
        return {"nodes": 0, "relationships": 0, "status": "skipped"}

    res = resolver or EntityResolver()
    nodes_written = 0
    rels_written = 0

    with driver.session() as session:
        for item in results:
            if not item.entities and not item.relationships:
                continue

            # 1. Merge Document node
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                ON CREATE SET d.filename = $filename, d.created_at = timestamp()
                ON MATCH SET d.filename = $filename
                """,
                {"doc_id": item.doc_id, "filename": filename},
            )

            # 2. Merge Entities and link to Document
            for ent in item.entities:
                canonical, aliases = res.resolve_entity(ent)
                session.run(
                    """
                    MERGE (e:Entity {canonical_name: $canonical, entity_type: $entity_type})
                    ON CREATE SET e.description = $desc, e.aliases = $aliases, e.created_at = timestamp()
                    ON MATCH SET e.aliases = [a IN coalesce(e.aliases, []) WHERE NOT a IN $aliases] + $aliases
                    WITH e
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (d)-[m:MENTIONS {chunk_id: $chunk_id}]->(e)
                    """,
                    {
                        "canonical": canonical,
                        "entity_type": ent.entity_type,
                        "desc": ent.description,
                        "aliases": aliases,
                        "doc_id": item.doc_id,
                        "chunk_id": item.chunk_id,
                    },
                )
                nodes_written += 1

            # 3. Merge Relationships between Entities
            for raw_rel in item.relationships:
                rel = res.resolve_relationship(raw_rel)

                # Ensure rel_type is in allowed whitelist
                rel_type = rel.rel_type if rel.rel_type in RELATIONSHIP_TYPES else "CONNECTS_TO"

                cypher = f"""
                MATCH (s:Entity {{canonical_name: $source, entity_type: $source_type}})
                MATCH (t:Entity {{canonical_name: $target, entity_type: $target_type}})
                MERGE (s)-[r:{rel_type} {{source_chunk_id: $chunk_id}}]->(t)
                ON CREATE SET r.confidence = $conf, r.doc_id = $doc_id, r.created_at = timestamp()
                """

                session.run(
                    cypher,
                    {
                        "source": rel.source_name,
                        "source_type": rel.source_type,
                        "target": rel.target_name,
                        "target_type": rel.target_type,
                        "conf": rel.confidence,
                        "doc_id": item.doc_id,
                        "chunk_id": item.chunk_id,
                    },
                )
                rels_written += 1

    logger.info(
        "Wrote graph for doc_id=%s: %d entity mentions, %d relationships",
        results[0].doc_id if results else "none",
        nodes_written,
        rels_written,
    )
    return {
        "nodes": nodes_written,
        "relationships": rels_written,
        "status": "success",
    }


def get_graph_stats() -> dict:
    """Return high-level counts and breakdown of the Neo4j knowledge graph."""
    driver = get_driver()
    if driver is None:
        return {"error": "Neo4j unavailable", "node_count": 0, "edge_count": 0}

    with driver.session() as session:
        node_res = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()
        node_count = node_res["count"] if node_res else 0

        edge_res = session.run(
            "MATCH ()-[r]->() WHERE type(r) <> 'MENTIONS' RETURN count(r) AS count"
        ).single()
        edge_count = edge_res["count"] if edge_res else 0

        # Type breakdown
        type_res = session.run(
            "MATCH (e:Entity) RETURN e.entity_type AS type, count(e) AS count ORDER BY count DESC LIMIT 10"
        )
        type_breakdown = {record["type"]: record["count"] for record in type_res}

        # Top connected entities
        top_res = session.run(
            """
            MATCH (e:Entity)-[r]-()
            WHERE type(r) <> 'MENTIONS'
            RETURN e.canonical_name AS name, e.entity_type AS type, count(r) AS degree
            ORDER BY degree DESC
            LIMIT 5
            """
        )
        top_entities = [
            {"name": record["name"], "type": record["type"], "degree": record["degree"]}
            for record in top_res
        ]

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "type_breakdown": type_breakdown,
            "top_entities": top_entities,
        }


def query_entity_neighborhood(entity_name: str, depth: int = 1) -> list[dict]:
    """
    Find 1-hop or 2-hop neighborhood around an entity name for graph retrieval.
    Returns triples: (source, relation, target, chunk_id).
    """
    driver = get_driver()
    if driver is None:
        return []

    depth = max(1, min(2, depth))  # clamp between 1 and 2

    with driver.session() as session:
        query = f"""
        MATCH (s:Entity)
        WHERE toLower(s.canonical_name) = toLower($name)
           OR ANY(alias IN s.aliases WHERE toLower(alias) = toLower($name))
        MATCH (s)-[r]-(t:Entity)
        WHERE type(r) <> 'MENTIONS'
        RETURN s.canonical_name AS source,
               s.entity_type AS source_type,
               type(r) AS relation,
               t.canonical_name AS target,
               t.entity_type AS target_type,
               r.source_chunk_id AS source_chunk_id
        LIMIT 25
        """
        records = session.run(query, {"name": entity_name})
        results = []
        for rec in records:
            results.append(
                {
                    "source": rec["source"],
                    "source_type": rec["source_type"],
                    "relation": rec["relation"],
                    "target": rec["target"],
                    "target_type": rec["target_type"],
                    "source_chunk_id": rec["source_chunk_id"],
                }
            )
        return results


def clear_graph_for_document(doc_id: int) -> None:
    """Remove all relationships and document nodes for a specific doc_id."""
    driver = get_driver()
    if driver is None:
        return

    with driver.session() as session:
        session.run(
            """
            MATCH (d:Document {doc_id: $doc_id})
            OPTIONAL MATCH (d)-[m:MENTIONS]->()
            DELETE m, d
            """,
            {"doc_id": doc_id},
        )
        session.run(
            """
            MATCH ()-[r {doc_id: $doc_id}]->()
            DELETE r
            """,
            {"doc_id": doc_id},
        )
