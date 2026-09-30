"""Extract and validate answer citations against retrieved PostgreSQL chunk IDs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_CITATION_PATTERN = re.compile(r"\[\s*chunk_id\s*[:=]\s*(\d+)\s*\]", re.IGNORECASE)
_NON_CLAIM_PATTERN = re.compile(
    r"^(?:i (?:do not|don't) know|the (?:retrieved )?evidence (?:does not|doesn't) contain|"
    r"no (?:relevant )?evidence|no verified answer|i couldn't validate a generated summary|"
    r"these are the retrieved source sentences|i cannot answer|unable to answer)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationValidation:
    valid: bool
    citation_ids: tuple[int, ...]
    invalid_ids: tuple[int, ...]
    uncited_claims: tuple[str, ...]


def extract_citations(answer: str) -> list[int]:
    """Return chunk IDs cited using ``[chunk_id: N]`` or ``[chunk_id=N]``."""
    return [int(match) for match in _CITATION_PATTERN.findall(answer)]


def validate_citations(answer: str, retrieved_chunk_ids: Iterable[int]) -> CitationValidation:
    """Require all cited IDs to be retrieved and every factual sentence to cite one."""
    retrieved_ids = {int(chunk_id) for chunk_id in retrieved_chunk_ids}
    citation_ids = tuple(extract_citations(answer))
    invalid_ids = tuple(dict.fromkeys(chunk_id for chunk_id in citation_ids if chunk_id not in retrieved_ids))

    fragments = re.split(
        r"(?<=[.!?])\s+(?!\[?\s*chunk_id\s*[:=])|\]\s+(?=[A-Z0-9*#-])|\n+",
        answer,
    )
    uncited_claims = []
    for fragment in fragments:
        has_citation = bool(_CITATION_PATTERN.search(fragment))
        claim = _CITATION_PATTERN.sub("", fragment)
        claim = re.sub(r"^[\s>*-]+", "", claim).strip()
        if not claim or claim.startswith("#") or _NON_CLAIM_PATTERN.match(claim):
            continue
        if not has_citation:
            uncited_claims.append(claim)

    return CitationValidation(
        valid=not invalid_ids and not uncited_claims,
        citation_ids=citation_ids,
        invalid_ids=invalid_ids,
        uncited_claims=tuple(uncited_claims),
    )