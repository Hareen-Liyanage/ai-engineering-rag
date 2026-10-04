"""PII scrubber for generated answer text.

Replaces common PII patterns in LLM output with safe placeholders so that
personally identifiable information from ingested documents cannot leak into
the displayed answer or the telemetry log.

All substitutions are applied with compiled regex for performance; no external
dependencies are required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Pattern registry  (label, compiled regex, replacement placeholder)
# ---------------------------------------------------------------------------
_PII_RULES: list[tuple[str, re.Pattern, str]] = [
    # E-mail addresses
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL REDACTED]",
    ),
    # US phone numbers — many formats: (555) 867-5309 / 555-867-5309 / +1 555 867 5309
    (
        "phone",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?1[\s\-.]?)?"
            r"(?:\(?\d{3}\)?[\s\-.]?)"
            r"\d{3}[\s\-.]?\d{4}"
            r"(?!\d)"
        ),
        "[PHONE REDACTED]",
    ),
    # US Social Security Numbers — NNN-NN-NNNN or NNN NN NNNN
    (
        "ssn",
        re.compile(r"\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b"),
        "[SSN REDACTED]",
    ),
    # Credit / debit card numbers — 13–19 consecutive digits, possibly dash/space separated
    (
        "credit_card",
        re.compile(r"\b(?:\d[\s\-]?){13,19}\b"),
        "[CARD REDACTED]",
    ),
    # IPv4 addresses (can carry user context in logs)
    (
        "ipv4",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[IP REDACTED]",
    ),
    # Passport numbers — letter(s) followed by 6–9 digits
    (
        "passport",
        re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
        "[PASSPORT REDACTED]",
    ),
    # UK National Insurance numbers
    (
        "uk_ni",
        re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D ]\b", re.I),
        "[NI REDACTED]",
    ),
]


@dataclass(frozen=True)
class ScrubResult:
    text: str           # cleaned answer text
    scrubbed: bool      # True if any PII was replaced
    types_found: tuple[str, ...]  # labels of PII types that were redacted


def scrub_answer(text: str) -> ScrubResult:
    """Scan *text* for PII patterns and replace each match with a safe placeholder.

    Returns a :class:`ScrubResult` containing the cleaned text and metadata
    about which PII types were detected.
    """
    found_types: list[str] = []
    current = text

    for label, pattern, placeholder in _PII_RULES:
        # Count matches before substitution to know whether this rule fired
        if pattern.search(current):
            found_types.append(label)
            current = pattern.sub(placeholder, current)

    return ScrubResult(
        text=current,
        scrubbed=bool(found_types),
        types_found=tuple(found_types),
    )
