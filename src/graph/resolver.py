"""
src/graph/resolver.py — Entity normalization, alias management, and deduplication
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

from src.graph.ontology import ExtractedEntity, ExtractedRelationship

logger = logging.getLogger(__name__)

# Common legal/corporate suffixes to strip for canonical resolution
_CORP_SUFFIXES = re.compile(
    r"\b(Inc\.?|LLC\.?|Corp\.?|Corporation|Ltd\.?|Limited|Co\.?|Company)\b",
    re.IGNORECASE,
)

# Common known alias mappings (variant -> canonical)
_KNOWN_ALIASES: dict[str, str] = {
    "postgres": "PostgreSQL",
    "postgresql database": "PostgreSQL",
    "neo4j desktop": "Neo4j",
    "neo4j graph": "Neo4j",
    "neo4j database": "Neo4j",
    "ollama server": "Ollama",
    "chatgpt": "OpenAI ChatGPT",
    "claude": "Anthropic Claude",
}


def normalize_entity_name(name: str) -> str:
    """Normalize entity name into clean canonical format."""
    cleaned = name.strip().strip("'\"`.,;:()")
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Check known alias table (case-insensitive)
    lower = cleaned.lower()
    if lower in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[lower]

    # Strip trailing corporate suffixes for organizations
    stripped = _CORP_SUFFIXES.sub("", cleaned).strip().strip(" ,")
    if stripped and len(stripped) > 2:
        return stripped

    return cleaned


class EntityResolver:
    """
    Maintains in-memory canonical registry to resolve entity names and aliases
    before writing to Neo4j.
    """

    def __init__(self) -> None:
        # (entity_type, normalized_lower) -> canonical_name
        self._lookup: dict[tuple[str, str], str] = {}
        # canonical_name -> set of seen aliases
        self._aliases: dict[str, set[str]] = {}

    def resolve_entity(self, entity: ExtractedEntity) -> tuple[str, list[str]]:
        """
        Resolve an entity to its canonical name and list of registered aliases.
        Returns (canonical_name, aliases_list).
        """
        raw_name = entity.name.strip()
        canonical = normalize_entity_name(raw_name)
        key = (entity.entity_type, canonical.lower())

        if key in self._lookup:
            canonical = self._lookup[key]
        else:
            self._lookup[key] = canonical

        # Record alias if different from canonical
        if canonical not in self._aliases:
            self._aliases[canonical] = set()

        if raw_name != canonical:
            self._aliases[canonical].add(raw_name)

        return canonical, sorted(list(self._aliases[canonical]))

    def resolve_relationship(
        self,
        rel: ExtractedRelationship,
    ) -> ExtractedRelationship:
        """Resolve both source and target entities in a relationship."""
        norm_source = normalize_entity_name(rel.source_name)
        norm_target = normalize_entity_name(rel.target_name)

        source_key = (rel.source_type, norm_source.lower())
        target_key = (rel.target_type, norm_target.lower())

        resolved_source = self._lookup.get(source_key, norm_source)
        resolved_target = self._lookup.get(target_key, norm_target)

        return ExtractedRelationship(
            source_name=resolved_source,
            source_type=rel.source_type,
            rel_type=rel.rel_type,
            target_name=resolved_target,
            target_type=rel.target_type,
            confidence=rel.confidence,
        )
