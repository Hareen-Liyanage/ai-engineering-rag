"""Exact-match and LLM-judge scoring for benchmark answers."""
from __future__ import annotations

import json
import re
import string
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


def normalize_answer(answer: str) -> str:
    """Normalize casing, punctuation, and whitespace for exact match."""
    lowered = answer.casefold()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    return " ".join(without_punctuation.split())


def exact_match(expected: str, actual: str) -> bool:
    return normalize_answer(expected) == normalize_answer(actual)


def judge_answer(
    question: str,
    expected_answer: str,
    actual_answer: str,
    llm: Any,
) -> dict[str, Any]:
    """Ask an LLM to judge semantic correctness and return structured output."""
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strict benchmark evaluator. Judge whether the actual answer "
                    "correctly answers the question and matches the reference meaning. "
                    "For an out-of-scope item whose reference says REFUSE, pass only if "
                    "the actual answer clearly refuses or says the evidence is insufficient. "
                    'Return only JSON: {"correct": true|false, "score": 0|1, "reason": "..."}.'
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {question}\nReference answer: {expected_answer}\n"
                    f"Actual answer: {actual_answer}"
                )
            ),
        ]
    )
    raw = str(response.content).strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    payload = fenced.group(1).strip() if fenced else raw
    parsed = json.loads(payload)
    correct = bool(parsed["correct"])
    score = float(parsed.get("score", 1.0 if correct else 0.0))
    return {
        "correct": correct,
        "score": max(0.0, min(1.0, score)),
        "reason": str(parsed.get("reason", "")),
    }
