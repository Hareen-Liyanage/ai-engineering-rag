"""Guardrails orchestrator — single entry point for Phase 6 safety checks.

Call :func:`run_guardrails` from the retrieval pipeline to:

1. **Pre-retrieval gate** — :mod:`topic_guard` checks the *question* for
   jailbreak attempts, PII-exfiltration requests, off-topic prompts, and
   harmful content.  A blocked question is refused immediately without any
   database or LLM call.

2. **Post-generation gate** — :mod:`groundedness` verifies that every cited
   sentence in the answer is lexically grounded in its referenced chunk.
   Ungrounded answers are flagged so the caller can fall back to extractive
   evidence.

3. **PII scrubber** — :mod:`pii_scrubber` removes PII patterns (emails, phone
   numbers, SSNs, credit cards, …) from the final answer text.

All guardrail signals are returned in :class:`GuardrailsResult` and are also
attached to the JSONL telemetry log by the router.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import settings
from src.guardrails.groundedness import GroundednessResult, check_groundedness
from src.guardrails.pii_scrubber import ScrubResult, scrub_answer
from src.guardrails.topic_guard import TopicGuardResult, check_question

logger = logging.getLogger(__name__)

_BLOCKED_ANSWER_TEMPLATE = (
    "⚠️ This question could not be answered. {reason}"
)


@dataclass(frozen=True)
class GuardrailsResult:
    # --- Pre-retrieval gate ---
    topic_guard: TopicGuardResult

    # --- Post-generation gate (None when question was blocked before generation) ---
    groundedness: GroundednessResult | None

    # --- PII scrubber ---
    scrub: ScrubResult | None

    # --- Final answer text (after all guardrails applied) ---
    answer: str

    @property
    def blocked(self) -> bool:
        """True if the question was refused by the topic guard."""
        return self.topic_guard.blocked

    @property
    def grounded(self) -> bool:
        """True if the answer passed the groundedness check (or was blocked)."""
        if self.groundedness is None:
            return True  # blocked before generation — not applicable
        return self.groundedness.grounded

    @property
    def pii_scrubbed(self) -> bool:
        return self.scrub is not None and self.scrub.scrubbed


def run_guardrails(
    question: str,
    answer: str | None = None,
    context: str = "",
    retrieved_chunk_ids: list[int] | None = None,
) -> GuardrailsResult:
    """Run all guardrail checks and return a :class:`GuardrailsResult`.

    Parameters
    ----------
    question:
        The raw user question.
    answer:
        The generated answer text.  Pass ``None`` when calling as a
        pre-retrieval gate only (the function will skip post-generation checks).
    context:
        The assembled context string from :func:`assembler.assemble_context`.
    retrieved_chunk_ids:
        IDs of chunks that were retrieved (used by groundedness check).

    Returns
    -------
    GuardrailsResult
        Contains all guardrail signals and the final (possibly mutated) answer.
    """
    if not settings.guardrails_enabled:
        # Guardrails disabled — pass through everything unchanged
        return GuardrailsResult(
            topic_guard=TopicGuardResult(blocked=False, reason=None, category=None),
            groundedness=None,
            scrub=None,
            answer=answer or "",
        )

    # ── 1. Pre-retrieval gate ────────────────────────────────────────────────
    topic_result = check_question(question)
    if topic_result.blocked:
        logger.warning(
            "guardrails_blocked category=%s question=%r",
            topic_result.category,
            question[:120],
        )
        return GuardrailsResult(
            topic_guard=topic_result,
            groundedness=None,
            scrub=None,
            answer=_BLOCKED_ANSWER_TEMPLATE.format(reason=topic_result.reason),
        )

    # No answer provided — caller only wanted the pre-retrieval gate
    if answer is None:
        return GuardrailsResult(
            topic_guard=topic_result,
            groundedness=None,
            scrub=None,
            answer="",
        )

    # ── 2. Post-generation groundedness check ────────────────────────────────
    groundedness_result = check_groundedness(answer, context)
    if not groundedness_result.grounded:
        logger.warning(
            "guardrails_ungrounded mean_score=%.3f ungrounded_count=%d",
            groundedness_result.mean_score,
            len(groundedness_result.ungrounded_sentences),
        )

    # ── 3. PII scrubber ──────────────────────────────────────────────────────
    scrub_result = scrub_answer(answer)
    if scrub_result.scrubbed:
        logger.warning(
            "guardrails_pii_scrubbed types=%s",
            scrub_result.types_found,
        )

    final_answer = scrub_result.text

    return GuardrailsResult(
        topic_guard=topic_result,
        groundedness=groundedness_result,
        scrub=scrub_result,
        answer=final_answer,
    )
