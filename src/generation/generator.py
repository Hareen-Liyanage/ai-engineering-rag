"""Ollama-first grounded answer generation with citation retries."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.messages import HumanMessage

from src.config import settings
from src.generation.citation_validator import CitationValidation, validate_citations
from src.retrieval.assembler import build_messages

logger = logging.getLogger(__name__)

_MAX_CITATION_RETRIES = 2
_REJECTED_ANSWER = "No verified answer was found in the retrieved evidence."
_GENERATION_ERROR = "I couldn't generate an answer because the language model was unavailable."
_FALLBACK_UNSET = object()


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    attempts: int
    retries: int
    valid: bool
    citation_ids: tuple[int, ...]
    invalid_ids: tuple[int, ...]
    provider: str
    uncited_claim_count: int = 0
    extractive_fallback: bool = False


def _get_ollama_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=0,
    )


def _get_fallback_llm():
    if not settings.llm_fallback_enabled:
        return None
    if settings.llm_fallback_provider == "anthropic" and settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )
    if settings.llm_fallback_provider == "openai" and settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    logger.warning(
        "LLM fallback is enabled, but provider '%s' is unsupported or has no API key",
        settings.llm_fallback_provider,
    )
    return None


def _retry_message(
    question: str,
    context: str,
    validation: CitationValidation,
) -> list:
    messages = build_messages(question, context)
    problems = []
    if validation.invalid_ids:
        problems.append(f"These chunk IDs were not retrieved: {list(validation.invalid_ids)}.")
    if validation.uncited_claims:
        problems.append(f"These claims lacked citations: {list(validation.uncited_claims)}.")
    messages.append(
        HumanMessage(
            content=(
                "Rewrite the complete answer. Fix the citation problems below, cite every factual sentence "
                "with [chunk_id: N], and use only IDs present in the evidence.\n"
                + "\n".join(problems)
            )
        )
    )
    return messages


def _extractive_fallback(context: str, retrieved_ids: set[int]) -> str:
    """Return source sentences verbatim with their retrieved chunk IDs."""
    excerpts: list[str] = []
    for section in context.split("\n\n---\n\n"):
        header, separator, body = section.partition("\n")
        if not separator:
            continue
        match = re.search(r"\bchunk_id\s*:\s*(\d+)\b", header, re.IGNORECASE)
        if not match or int(match.group(1)) not in retrieved_ids:
            continue
        chunk_id = int(match.group(1))
        sentences = re.split(r"(?<=[.!?])\s+|\n+", body.strip())
        for sentence in sentences:
            sentence = sentence.strip().lstrip("-•* ")
            if sentence:
                excerpts.append(f"{sentence} [chunk_id: {chunk_id}]")
            if len(excerpts) == 3:
                break
        if len(excerpts) == 3:
            break
    if not excerpts:
        return _REJECTED_ANSWER
    return "I couldn't validate a generated summary; these are the retrieved source sentences:\n" + "\n".join(
        excerpts
    )


def generate_answer(
    question: str,
    context: str,
    retrieved_chunk_ids: Iterable[int],
    llm: Any | None = None,
    fallback_llm: Any = _FALLBACK_UNSET,
    max_retries: int = _MAX_CITATION_RETRIES,
) -> GenerationResult:
    """Generate a cited answer, retrying invalid citations at most twice."""
    retrieved_ids = tuple(dict.fromkeys(int(chunk_id) for chunk_id in retrieved_chunk_ids))
    retries_allowed = max(0, min(_MAX_CITATION_RETRIES, max_retries))
    try:
        model = llm or _get_ollama_llm()
    except Exception:
        logger.exception("Unable to initialize Ollama answer generation")
        return GenerationResult(_GENERATION_ERROR, 0, 0, False, (), (), "ollama")

    provider = "injected" if llm is not None else "ollama"
    if fallback_llm is _FALLBACK_UNSET:
        try:
            fallback_llm = _get_fallback_llm()
        except Exception:
            logger.exception("Unable to initialize configured LLM fallback")
            fallback_llm = None

    attempts = 0
    retries = 0
    fallback_used = False
    previous_validation: CitationValidation | None = None

    while True:
        attempts += 1
        messages = (
            _retry_message(question, context, previous_validation)
            if previous_validation is not None
            else build_messages(question, context)
        )
        try:
            response = model.invoke(messages)
        except Exception:
            logger.exception("Answer generation failed using %s", provider)
            if fallback_llm is not None and not fallback_used:
                model = fallback_llm
                provider = settings.llm_fallback_provider
                fallback_used = True
                continue
            return GenerationResult(_GENERATION_ERROR, attempts, retries, False, (), (), provider)

        answer = str(response.content).strip()
        validation = validate_citations(answer, retrieved_ids)
        if validation.valid:
            return GenerationResult(
                answer,
                attempts,
                retries,
                True,
                validation.citation_ids,
                validation.invalid_ids,
                provider,
                len(validation.uncited_claims),
            )

        previous_validation = validation
        if retries >= retries_allowed:
            logger.warning(
                "Rejecting answer after %d citation retries; invalid_ids=%s uncited_claims=%d",
                retries,
                validation.invalid_ids,
                len(validation.uncited_claims),
            )
            fallback_answer = _extractive_fallback(context, set(retrieved_ids))
            extractive_fallback = fallback_answer != _REJECTED_ANSWER
            fallback_validation = validate_citations(fallback_answer, retrieved_ids)
            return GenerationResult(
                fallback_answer,
                attempts,
                retries,
                fallback_validation.valid,
                fallback_validation.citation_ids,
                fallback_validation.invalid_ids,
                provider,
                len(fallback_validation.uncited_claims),
                extractive_fallback,
            )
        retries += 1