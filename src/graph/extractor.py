"""
src/graph/extractor.py — LLM-driven entity and relationship extraction
"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings
from src.graph.ontology import (
    ENTITY_TYPES,
    RELATIONSHIP_TYPES,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = f"""You are an expert knowledge graph extraction engine.
Given an excerpt of text, extract the key entities and the relationships between them.

Allowed Entity Types:
{', '.join(ENTITY_TYPES)}

Allowed Relationship Types:
{', '.join(RELATIONSHIP_TYPES)}

Rules:
1. ONLY extract explicitly stated entities and relationships. Do not hallucinate.
2. Names must be concise and canonical (e.g. 'PostgreSQL' not 'the relational database PostgreSQL').
3. You MUST respond with ONLY a valid JSON object in this exact schema, with no preamble:
{{
  "entities": [
    {{"name": "Entity Name", "entity_type": "One of allowed types", "description": "Brief description"}}
  ],
  "relationships": [
    {{"source": "Source Name", "source_type": "Source Entity Type", "relation": "One of allowed relations", "target": "Target Name", "target_type": "Target Entity Type", "confidence": 0.95}}
  ]
}}
"""


def _get_llm():
    """Return the configured chat model."""
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )
    elif settings.llm_provider == "openai" and settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    else:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            temperature=0,
        )


def _parse_json_response(raw_text: str) -> dict:
    """Safely extract and parse JSON from model output."""
    raw = raw_text.strip()

    # Remove markdown code fence if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if match:
        raw = match.group(1).strip()

    # If it starts before the first '{', slice to first '{' and last '}'
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1:
        raw = raw[first_brace : last_brace + 1]

    # Clean trailing commas before closing braces/brackets
    raw = re.sub(r",\s*([\]}])", r"\1", raw)

    return json.loads(raw)


def extract_knowledge_from_chunk(
    chunk_text: str,
    chunk_id: int,
    doc_id: int,
) -> ExtractionResult:
    """
    Run LLM extraction over chunk_text and return structured ExtractionResult.
    Never raises — returns an empty ExtractionResult on failure.
    """
    if not chunk_text.strip():
        return ExtractionResult(chunk_id=chunk_id, doc_id=doc_id)

    llm = _get_llm()
    prompt = f"Text to extract from:\n{chunk_text}"

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        content_str = str(response.content)
        parsed = _parse_json_response(content_str)

        entities: list[ExtractedEntity] = []
        for e in parsed.get("entities", []):
            if isinstance(e, dict) and e.get("name") and e.get("entity_type"):
                entities.append(
                    ExtractedEntity(
                        name=str(e["name"]),
                        entity_type=str(e["entity_type"]),
                        description=str(e.get("description", "")),
                    )
                )

        relationships: list[ExtractedRelationship] = []
        for r in parsed.get("relationships", []):
            if isinstance(r, dict) and r.get("source") and r.get("target") and r.get("relation"):
                relationships.append(
                    ExtractedRelationship(
                        source_name=str(r["source"]),
                        source_type=str(r.get("source_type", "Concept")),
                        rel_type=str(r["relation"]),
                        target_name=str(r["target"]),
                        target_type=str(r.get("target_type", "Concept")),
                        confidence=float(r.get("confidence", 1.0)),
                    )
                )

        logger.debug(
            "Extracted %d entities, %d relationships from chunk %d",
            len(entities),
            len(relationships),
            chunk_id,
        )
        return ExtractionResult(
            chunk_id=chunk_id,
            doc_id=doc_id,
            entities=entities,
            relationships=relationships,
        )

    except Exception as exc:
        logger.warning("Extraction failed for chunk %d: %s", chunk_id, exc)
        return ExtractionResult(chunk_id=chunk_id, doc_id=doc_id)
