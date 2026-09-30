"""
test_phase3.py — End-to-end test of Phase 3 Knowledge Graph construction
"""
from src.db import neo4j_conn, postgres
from src.graph.ontology import ExtractedEntity, ExtractedRelationship, ExtractionResult
from src.graph.extractor import extract_knowledge_from_chunk
from src.graph.resolver import EntityResolver, normalize_entity_name
from src.graph.writer import (
    write_extraction_to_graph,
    get_graph_stats,
    query_entity_neighborhood,
    clear_graph_for_document,
)
from src.ingestion.ingest import ingest_pdf


def main():
    print("=== 1. Testing Neo4j Schema Init ===")
    neo4j_conn.init_schema()
    postgres.init_schema()
    print("Neo4j and PostgreSQL schemas initialized.")

    print("\n=== 2. Testing Entity Resolution ===")
    resolver = EntityResolver()
    e1 = ExtractedEntity(name="PostgreSQL", entity_type="Technology")
    e2 = ExtractedEntity(name="Postgres", entity_type="Technology")
    e3 = ExtractedEntity(name="Neo4j Inc.", entity_type="Organization")

    c1, a1 = resolver.resolve_entity(e1)
    c2, a2 = resolver.resolve_entity(e2)
    c3, a3 = resolver.resolve_entity(e3)

    print(f"  '{e1.name}' -> canonical: '{c1}', aliases: {a1}")
    print(f"  '{e2.name}' -> canonical: '{c2}', aliases: {a2}")
    print(f"  '{e3.name}' -> canonical: '{c3}', aliases: {a3}")

    assert c1 == c2 == "PostgreSQL", f"Expected canonical 'PostgreSQL', got '{c1}' and '{c2}'"
    assert "Postgres" in a2 or "postgres" in str(a2).lower()
    assert c3 == "Neo4j"

    print("\n=== 3. Testing Graph Writer (Idempotent MERGE) ===")
    # Clear any previous test doc 999
    clear_graph_for_document(doc_id=999)

    test_res = ExtractionResult(
        chunk_id=101,
        doc_id=999,
        entities=[
            ExtractedEntity(name="Hareen Liyanage", entity_type="Person", description="AI Engineer"),
            ExtractedEntity(name="Knowledge Graph RAG", entity_type="Project", description="Hybrid RAG system"),
            ExtractedEntity(name="Neo4j", entity_type="Technology", description="Graph database"),
            ExtractedEntity(name="PostgreSQL", entity_type="Technology", description="Relational database"),
        ],
        relationships=[
            ExtractedRelationship(
                source_name="Hareen Liyanage",
                source_type="Person",
                rel_type="DEVELOPS",
                target_name="Knowledge Graph RAG",
                target_type="Project",
            ),
            ExtractedRelationship(
                source_name="Knowledge Graph RAG",
                source_type="Project",
                rel_type="USES",
                target_name="Neo4j",
                target_type="Technology",
            ),
            ExtractedRelationship(
                source_name="Knowledge Graph RAG",
                source_type="Project",
                rel_type="USES",
                target_name="PostgreSQL",
                target_type="Technology",
            ),
        ],
    )

    write_res = write_extraction_to_graph([test_res], filename="test_doc_999.pdf", resolver=resolver)
    print(f"Write result: {write_res}")
    assert write_res["status"] == "success"

    print("\n=== 4. Testing Graph Queries & Neighborhood Traversal ===")
    stats = get_graph_stats()
    print(f"Graph stats: {stats}")
    assert stats["node_count"] >= 4
    assert stats["edge_count"] >= 3

    neighborhood = query_entity_neighborhood("Knowledge Graph RAG")
    print(f"Neighborhood for 'Knowledge Graph RAG' ({len(neighborhood)} connections):")
    for link in neighborhood:
        print(f"  {link['source']} -[{link['relation']}]-> {link['target']} (source_chunk: {link['source_chunk_id']})")
    assert len(neighborhood) >= 3

    print("\n=== 5. Testing LLM Extraction on Real Sample Text ===")
    sample_text = (
        "Alice Johnson is a principal researcher at DeepMind in London. "
        "She leads Project Gemini, an initiative developing multi-modal artificial intelligence systems. "
        "Project Gemini relies on PyTorch and custom TPU hardware."
    )
    extracted = extract_knowledge_from_chunk(sample_text, chunk_id=202, doc_id=888)
    print(f"LLM Extracted {len(extracted.entities)} entities and {len(extracted.relationships)} relationships:")
    for ent in extracted.entities:
        print(f"  Entity: [{ent.entity_type}] {ent.name} — {ent.description}")
    for rel in extracted.relationships:
        print(f"  Rel: ({rel.source_name}) -[{rel.rel_type}]-> ({rel.target_name})")

    if extracted.entities:
        write_extraction_to_graph([extracted], filename="deepmind_memo.pdf")
        print("Successfully wrote LLM extracted triples to Neo4j!")

    print("\n>>> Phase 3 Knowledge Graph test PASSED successfully! <<<")


if __name__ == "__main__":
    main()
