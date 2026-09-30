"""
src/graph/ontology.py — Domain ontology and structured extraction models
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# ── Allowed Ontology ──────────────────────────────────────────────────────────

ENTITY_TYPES = (
    "Person",
    "Organization",
    "Technology",
    "Location",
    "Product",
    "Concept",
    "Event",
    "Project",
)

RELATIONSHIP_TYPES = (
    "USES",
    "DEVELOPS",
    "MANAGES",
    "PART_OF",
    "AFFILIATED_WITH",
    "LOCATED_IN",
    "CONNECTS_TO",
    "DEPENDS_ON",
    "PRODUCES",
    "BASED_ON",
    "COMPETES_WITH",
    "MENTIONS",
)


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    description: str = ""

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        # Fallback to Concept if unrecognized
        if self.entity_type not in ENTITY_TYPES:
            for et in ENTITY_TYPES:
                if et.lower() == self.entity_type.lower():
                    self.entity_type = et
                    return
            self.entity_type = "Concept"


@dataclass
class ExtractedRelationship:
    source_name: str
    source_type: str
    rel_type: str
    target_name: str
    target_type: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.source_name = self.source_name.strip()
        self.target_name = self.target_name.strip()
        self.rel_type = self.rel_type.strip().upper().replace(" ", "_")

        # Normalize relationship type
        if self.rel_type not in RELATIONSHIP_TYPES:
            for rt in RELATIONSHIP_TYPES:
                if rt == self.rel_type:
                    self.rel_type = rt
                    return
            self.rel_type = "CONNECTS_TO"


@dataclass
class ExtractionResult:
    chunk_id: int
    doc_id: int
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
