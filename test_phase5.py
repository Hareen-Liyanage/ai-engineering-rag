"""Offline unit tests for grounded answer generation and citation validation."""
import unittest
from unittest.mock import patch

from src.generation.citation_validator import extract_citations, validate_citations
from src.generation.generator import generate_answer


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class _SequenceLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return _Response(response)


class Phase5Tests(unittest.TestCase):
    def test_extract_citations_accepts_colon_and_equals(self):
        self.assertEqual(extract_citations("Fact. [chunk_id: 12] [chunk_id=14]"), [12, 14])

    def test_validator_accepts_only_retrieved_chunk_ids(self):
        result = validate_citations("A supported fact. [chunk_id: 12]", [12, 13])
        self.assertTrue(result.valid)
        self.assertEqual(result.citation_ids, (12,))

    def test_validator_rejects_hallucinated_chunk_id(self):
        result = validate_citations("A fabricated fact. [chunk_id: 999]", [12])
        self.assertFalse(result.valid)
        self.assertEqual(result.invalid_ids, (999,))

    def test_validator_rejects_uncited_claim(self):
        result = validate_citations("An uncited factual claim.", [12])
        self.assertFalse(result.valid)
        self.assertEqual(result.uncited_claims, ("An uncited factual claim.",))

    def test_evidence_header_is_not_a_citation(self):
        result = validate_citations(
            "[VECTOR | report.pdf | Page 1 | chunk_id: 12 | Chunk index: 0]",
            [12],
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.citation_ids, ())

    def test_validator_allows_abstention_without_evidence(self):
        result = validate_citations("The retrieved evidence does not contain the answer.", [])
        self.assertTrue(result.valid)

    def test_invalid_citation_regenerates_once_and_accepts_valid_answer(self):
        llm = _SequenceLLM(
            [
                "Claim. [chunk_id: 999]",
                "Supported claim. [chunk_id: 12]",
            ]
        )
        result = generate_answer("Question?", "[chunk_id: 12] evidence", [12], llm=llm, fallback_llm=None)
        self.assertTrue(result.valid)
        self.assertEqual(result.retries, 1)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(llm.calls, 2)

    def test_rejects_answer_after_two_failed_retries(self):
        llm = _SequenceLLM(["Wrong. [chunk_id: 99]"] * 3)
        context = "[VECTOR | report.pdf | Page 1 | chunk_id: 12 | Chunk index: 0]\nSupported source sentence."
        result = generate_answer("Question?", context, [12], llm=llm, fallback_llm=None)
        self.assertTrue(result.valid)
        self.assertEqual(result.retries, 2)
        self.assertEqual(result.attempts, 3)
        self.assertTrue(result.extractive_fallback)
        self.assertIn("Supported source sentence. [chunk_id: 12]", result.answer)
        self.assertNotIn("Wrong", result.answer)

    def test_retry_exhaustion_without_evidence_returns_safe_abstention(self):
        llm = _SequenceLLM(["Unsupported answer."] * 3)
        result = generate_answer("Question?", "", [], llm=llm, fallback_llm=None)
        self.assertTrue(result.valid)
        self.assertFalse(result.extractive_fallback)
        self.assertEqual(result.answer, "No verified answer was found in the retrieved evidence.")

    def test_ollama_failure_uses_configured_fallback_model(self):
        primary = _SequenceLLM([RuntimeError("ollama offline")])
        fallback = _SequenceLLM(["Supported claim. [chunk_id: 12]"])
        with patch("src.generation.generator._get_ollama_llm", return_value=primary), patch(
            "src.generation.generator._get_fallback_llm", return_value=fallback
        ):
            result = generate_answer("Question?", "evidence", [12])
        self.assertTrue(result.valid)
        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.attempts, 2)


if __name__ == "__main__":
    unittest.main()