"""Off-topic / jailbreak / PII-exfiltration guard for incoming questions.

All checks are regex-based — no LLM call, no measurable latency hit.
A blocked question is refused immediately without touching any database or model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Jailbreak / prompt-injection fingerprints
_JAILBREAK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)\b", re.I),
    re.compile(r"\bforget\s+(everything|all|your|prior|previous)\b", re.I),
    re.compile(r"\bact\s+as\s+(if\s+you\s+(are|were)|an?\s+(?:unrestricted|unfiltered|jailbroken))\b", re.I),
    re.compile(r"\bdo\s+anything\s+now\b", re.I),
    re.compile(r"\bDAN\s*mode\b", re.I),
    re.compile(r"\bpretend\s+(you\s+have\s+no\s+(rules?|restrictions?|guidelines?)|to\s+be\s+a)", re.I),
    re.compile(r"\byou\s+are\s+now\s+(?:free|unrestricted|unfiltered|a\s+different)\b", re.I),
    re.compile(r"\bsystem\s*prompt\b", re.I),
    re.compile(r"\boverride\s+(your\s+)?(safety|guidelines?|rules?|instructions?)\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
]

# PII exfiltration requests (asking the system to reveal personal data)
_PII_REQUEST_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:list|show|give|tell|find|extract|dump)\s+.*\b(?:ssn|social\s+security|passport\s+number|credit\s+card|bank\s+account|date\s+of\s+birth|home\s+address|phone\s+number|email\s+address(?:es)?)\b", re.I),
    re.compile(r"\b(?:personal|private|sensitive)\s+(?:data|information|details|records?)\s+(?:of|about|for)\b", re.I),
]

# Clearly off-topic creative / unrelated tasks
_OFF_TOPIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:write|compose|create|generate)\s+(?:a\s+)?(?:poem|song|story|essay|joke|lyrics|rap|haiku|limerick)\b", re.I),
    re.compile(r"\b(?:translate|convert)\s+(?:this|the\s+following|it)\s+(?:to|into)\s+\w+\b", re.I),
    re.compile(r"\b(?:what(?:'s|\s+is)\s+(?:the\s+)?(?:weather|time|date|temperature))\b", re.I),
    re.compile(r"\b(?:book|reserve|order|buy|purchase)\s+(?:a\s+)?(?:ticket|flight|hotel|table|reservation)\b", re.I),
    re.compile(r"\b(?:play|stream|download)\s+(?:music|video|movie|game)\b", re.I),
    re.compile(r"\b(?:who\s+(?:is|was)\s+(?:the\s+)?(?:president|prime\s+minister|king|queen|ceo|founder)\s+of\s+(?!.*document|.*knowledge|.*evidence))\b", re.I),
    re.compile(r"\b(?:calculate|compute|solve)\s+\d+[\s\+\-\*\/\^]+\d+\b", re.I),
]

# Harmful / dangerous content requests
_HARMFUL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:how\s+to\s+(?:make|build|create|synthesize|produce))\s+(?:a\s+)?(?:bomb|weapon|poison|drug|meth|fentanyl|explosiv)\b", re.I),
    re.compile(r"\b(?:make|build|create|synthesize|produce)\s+(?:a\s+)?(?:bomb|weapon|poison|drug|meth|fentanyl|explosiv)\b", re.I),
    re.compile(r"\b(?:write|generate|produce|create)\s+(?:malware|ransomware|virus|exploit|payload|shellcode)\b", re.I),
    re.compile(r"\b(?:self[\s-]harm|suicide|kill\s+(?:myself|yourself|someone))\b", re.I),
    re.compile(r"\b(?:hack|crack|break\s+into|bypass|circumvent)\s+(?:a\s+)?(?:password|account|system|server|database|network)\b", re.I),
]


@dataclass(frozen=True)
class TopicGuardResult:
    blocked: bool
    reason: str | None   # None when not blocked
    category: str | None  # "jailbreak" | "pii_request" | "off_topic" | "harmful" | None


def check_question(question: str) -> TopicGuardResult:
    """Return a :class:`TopicGuardResult` for the given question.

    All pattern groups are tested in priority order.  The first match wins
    and the question is rejected immediately without LLM involvement.
    """
    q = question.strip()

    for pattern in _HARMFUL_PATTERNS:
        if pattern.search(q):
            return TopicGuardResult(
                blocked=True,
                reason="Your question requests harmful or dangerous content and cannot be answered.",
                category="harmful",
            )

    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(q):
            return TopicGuardResult(
                blocked=True,
                reason="Your question appears to contain a prompt-injection or jailbreak attempt and cannot be processed.",
                category="jailbreak",
            )

    for pattern in _PII_REQUEST_PATTERNS:
        if pattern.search(q):
            return TopicGuardResult(
                blocked=True,
                reason="Your question requests personal or sensitive data that this system will not retrieve or expose.",
                category="pii_request",
            )

    for pattern in _OFF_TOPIC_PATTERNS:
        if pattern.search(q):
            return TopicGuardResult(
                blocked=True,
                reason="Your question appears to be off-topic. This assistant only answers questions about the ingested knowledge base.",
                category="off_topic",
            )

    return TopicGuardResult(blocked=False, reason=None, category=None)
