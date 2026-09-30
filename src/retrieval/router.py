"""LLM query routing and end-to-end hybrid retrieval orchestration."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import settings
from src.generation.generator import GenerationResult, generate_answer
from src.retrieval import assembler, graph_search, vector_search

logger = logging.getLogger(__name__)


class RetrievalRoute(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    BOTH = "both"


@dataclass(frozen=True)
class RouteDecision:
    route: RetrievalRoute
    confidence: float
    fallback: bool = False


def _get_llm():
    """Return the chat model configured for this project."""
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key, temperature=0)
    if settings.llm_provider == "openai" and settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0)
    from langchain_ollama import ChatOllama

    return ChatOllama(base_url=settings.ollama_base_url, model=settings.ollama_chat_model, temperature=0)


def _parse_decision(raw: str) -> tuple[RetrievalRoute, float]:
    """Parse a route response, defaulting malformed model output to low confidence."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    payload = match.group(1).strip() if match else raw.strip()
    try:
        parsed = json.loads(payload)
        route = RetrievalRoute(str(parsed.get("route", "")).lower())
        confidence = float(parsed.get("confidence", 0.0))
        return route, max(0.0, min(1.0, confidence))
    except (ValueError, TypeError, json.JSONDecodeError):
        return RetrievalRoute.BOTH, 0.0


def classify_question(question: str, llm: Any | None = None) -> RouteDecision:
    """Classify a question and force hybrid retrieval when confidence is low."""
    if llm is None:
        llm = _get_llm()
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Route knowledge-base questions to exactly one retrieval path. Use graph for explicit "
                        "entity relationships, ownership, management, dependencies, or multi-hop questions; "
                        "use vector for definitions and explanations of concepts; use both when both evidence "
                        "types are useful. Return only JSON: {\"route\": \"vector|graph|both\", \"confidence\": 0.0}."
                    )
                ),
                HumanMessage(content=question),
            ]
        )
    except Exception:
        logger.exception("Question routing failed; falling back to both retrieval paths")
        return RouteDecision(RetrievalRoute.BOTH, 0.0, True)
    route, confidence = _parse_decision(str(response.content))
    fallback = confidence < settings.route_confidence_threshold
    return RouteDecision(RetrievalRoute.BOTH if fallback else route, confidence, fallback)


def _log_completed_query(
    question: str,
    decision: RouteDecision,
    elapsed_ms: float,
    answer: str,
    generation: GenerationResult,
) -> None:
    """Append one JSONL record per completed query for later evaluation."""
    record = {
        "question": question,
        "route": decision.route.value,
        "confidence": decision.confidence,
        "fallback": decision.fallback,
        "latency_ms": round(elapsed_ms, 2),
        "answer": answer,
        "citation_valid": generation.valid,
        "citation_retry_count": generation.retries,
        "generation_attempt_count": generation.attempts,
        "citation_retry_rate": round(generation.retries / max(generation.attempts, 1), 4),
        "citation_ids": list(generation.citation_ids),
        "invalid_citation_ids": list(generation.invalid_ids),
        "uncited_claim_count": generation.uncited_claim_count,
        "generation_provider": generation.provider,
    }
    logger.info("retrieval_decision %s", json.dumps(record, ensure_ascii=True))
    log_path = Path(__file__).resolve().parents[2] / "logs" / "retrieval.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        logger.exception("Unable to persist retrieval decision log")


def retrieve_and_answer(
    question: str,
    doc_ids: list[int] | None = None,
    documents: list[dict] | None = None,
    llm: Any | None = None,
    embedder: Any | None = None,
) -> dict:
    """Route, retrieve, assemble evidence, answer, and persist query telemetry."""
    started = time.perf_counter()
    model = llm
    decision = RouteDecision(RetrievalRoute.BOTH, 0.0, True)
    vector_results: list[dict] = []
    graph_results: list[dict] = []
    try:
        model = model or _get_llm()
        decision = classify_question(question, llm=model)
    except Exception:
        logger.exception("Unable to initialize the router; using both retrieval paths")

    if decision.route in (RetrievalRoute.VECTOR, RetrievalRoute.BOTH):
        try:
            vector_results = vector_search.search(question, doc_ids=doc_ids, embedder=embedder)
        except Exception:
            logger.exception("Vector retrieval failed")

    if decision.route in (RetrievalRoute.GRAPH, RetrievalRoute.BOTH):
        try:
            triples = graph_search.search(question, doc_ids=doc_ids, llm=model)
            graph_results = graph_search.triples_to_prose(triples)
        except Exception:
            logger.exception("Graph retrieval failed")

    context = assembler.assemble_context(vector_results, graph_results, documents)
    retrieved_chunk_ids = [
        int(match["id"])
        for match in vector_results
        if match.get("id") is not None
    ]
    retrieved_chunk_ids.extend(
        int(match["source_chunk_id"])
        for match in graph_results
        if match.get("source_chunk_id") is not None
    )
    generation = generate_answer(
        question,
        context,
        retrieved_chunk_ids,
        llm=llm,
    )
    answer = generation.answer

    elapsed_ms = (time.perf_counter() - started) * 1000
    _log_completed_query(question, decision, elapsed_ms, answer, generation)
    return {
        "answer": answer,
        "decision": decision,
        "latency_ms": elapsed_ms,
        "vector_results": vector_results,
        "graph_results": graph_results,
        "context": context,
        "generation": generation,
    }