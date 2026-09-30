"""
src/graph — Knowledge Graph extraction, entity resolution, and Neo4j writing
"""
from src.graph.ontology import (
    ENTITY_TYPES,
    RELATIONSHIP_TYPES,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from src.graph.extractor import extract_knowledge_from_chunk
from src.graph.resolver import EntityResolver
from src.graph.writer import (
    write_extraction_to_graph,
    get_graph_stats,
    query_entity_neighborhood,
    clear_graph_for_document,
)

__all__ = [
    "ENTITY_TYPES",
    "RELATIONSHIP_TYPES",
    "ExtractedEntity",
    "ExtractedRelationship",
    "ExtractionResult",
    "extract_knowledge_from_chunk",
    "EntityResolver",
    "write_extraction_to_graph",
    "get_graph_stats",
    "query_entity_neighborhood",
    "clear_graph_for_document",
]
