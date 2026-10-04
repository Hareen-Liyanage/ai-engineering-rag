"""Lexical groundedness checker for generated answers.

For every factual sentence in the answer, this module verifies that the key
content words in that sentence appear in the body of the chunk it cites.
A sentence that cites a valid chunk ID but asserts facts entirely absent from
that chunk is flagged as "ungrounded" — a form of citation laundering.

Design choices
--------------
* Purely lexical (token overlap) — no embeddings, no LLM call, no latency hit.
* Uses a light stopword filter so common function words don't dominate the score.
* Each sentence is scored against the *full text* of its cited chunk(s) from the
  retrieved context, not against the whole evidence block.
* The overall ``grounded`` flag requires *every* factual sentence to meet the
  configured minimum overlap threshold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can of in on at to for "
    "with by from into about between and or but not so if this that "
    "these those it its they them their there here what when where who "
    "which how all any both each few more most other some such no nor "
    "only own same than too very just s t don doesn doesn't wasn wasn't "
    "aren aren't wouldn wouldn't couldn couldn't shouldn shouldn't i we "
    "you he she as up out also only then than just over after before "
    "because although though while since during".split()
)

_CITATION_PATTERN = re.compile(r"\[\s*chunk_id\s*[:=]\s*(\d+)\s*\]", re.I)
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?!\[chunk_id\s*[:=]\s*\d+\])|(?<=\])\s+(?=[A-Z])",
    re.I,
)
_HEADER_PATTERN = re.compile(
    r"^\[(?:VECTOR|GRAPH)\s*\|[^\]]*chunk_id\s*[:=]\s*(\d+)[^\]]*\]$",
    re.I,
)
_NON_CLAIM_PATTERN = re.compile(
    r"^(?:i (?:do not|don't) know|the (?:retrieved )?evidence (?:does not|doesn't) contain|"
    r"no (?:relevant )?evidence|no verified answer|i couldn't validate|"
    r"these are the retrieved source sentences|i cannot answer|unable to answer)\b",
    re.I,
)


def _tokenize(text: str) -> set[str]:
    """Lower-case alphanumeric tokens (words and numbers) minus stopwords."""
    return {
        w
        for w in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if w not in _STOPWORDS and len(w) > 1
    }


def _parse_context_chunks(context: str) -> dict[int, str]:
    """Return a mapping of chunk_id → body text from the assembled context string."""
    chunks: dict[int, str] = {}
    for section in context.split("\n\n---\n\n"):
        header, _, body = section.partition("\n")
        m = _HEADER_PATTERN.match(header.strip())
        if m:
            chunk_id = int(m.group(1))
            chunks[chunk_id] = body.strip()
    return chunks


def _overlap_score(claim_tokens: set[str], evidence_tokens: set[str]) -> float:
    """Fraction of claim tokens that appear in the evidence token set."""
    if not claim_tokens:
        return 1.0  # vacuously grounded (empty claim)
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceGroundedness:
    sentence: str
    cited_ids: tuple[int, ...]
    overlap_score: float   # highest overlap across all cited chunks
    grounded: bool


@dataclass(frozen=True)
class GroundednessResult:
    grounded: bool                              # True iff all factual sentences are grounded
    mean_score: float                           # average overlap across factual sentences
    ungrounded_sentences: tuple[str, ...]       # sentences that failed the threshold
    sentence_details: tuple[SentenceGroundedness, ...]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def check_groundedness(answer: str, context: str) -> GroundednessResult:
    """Verify that each cited sentence is grounded in its referenced chunk.

    Parameters
    ----------
    answer:
        The generated answer text (may contain ``[chunk_id: N]`` citations).
    context:
        The assembled context string produced by :func:`assembler.assemble_context`.
    """
    threshold = settings.groundedness_min_overlap
    chunk_texts = _parse_context_chunks(context)
    # Pre-tokenize every chunk once
    chunk_token_sets: dict[int, set[str]] = {
        cid: _tokenize(text) for cid, text in chunk_texts.items()
    }

    sentences = _SENTENCE_SPLIT.split(answer)
    sentence_details: list[SentenceGroundedness] = []
    ungrounded: list[str] = []
    scores: list[float] = []

    for raw_sentence in sentences:
        sentence = raw_sentence.strip()
        if not sentence:
            continue

        # Strip citation markers to get the pure claim text
        clean = _CITATION_PATTERN.sub("", sentence).strip()
        clean = re.sub(r"^[\s>*\-]+", "", clean)

        # Skip non-claim lines (abstentions, headers, empty)
        if not clean or clean.startswith("#") or _NON_CLAIM_PATTERN.match(clean):
            continue

        cited_ids = tuple(int(m) for m in _CITATION_PATTERN.findall(sentence))

        # Sentences without any citation cannot be assessed for groundedness;
        # they will be caught by the Phase 5 citation validator instead.
        if not cited_ids:
            continue

        claim_tokens = _tokenize(clean)
        best_score = 0.0
        for cid in cited_ids:
            if cid in chunk_token_sets:
                score = _overlap_score(claim_tokens, chunk_token_sets[cid])
                best_score = max(best_score, score)
            # If the cited chunk ID isn't in our parsed context (e.g. graph
            # result without a context block), treat it as maximally grounded
            # to avoid false positives — Phase 5 already handles invalid IDs.
            else:
                best_score = max(best_score, 1.0)

        is_grounded = best_score >= threshold
        scores.append(best_score)
        if not is_grounded:
            ungrounded.append(clean)

        sentence_details.append(
            SentenceGroundedness(
                sentence=sentence,
                cited_ids=cited_ids,
                overlap_score=round(best_score, 4),
                grounded=is_grounded,
            )
        )

    mean_score = sum(scores) / len(scores) if scores else 1.0
    overall_grounded = len(ungrounded) == 0

    return GroundednessResult(
        grounded=overall_grounded,
        mean_score=round(mean_score, 4),
        ungrounded_sentences=tuple(ungrounded),
        sentence_details=tuple(sentence_details),
    )
