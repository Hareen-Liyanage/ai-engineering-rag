"""Offline unit tests for evaluation scoring."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.eval import runner
from src.eval.scorer import exact_match, judge_answer


class ScorerTests(unittest.TestCase):
    def test_exact_match_ignores_case_punctuation_and_whitespace(self):
        self.assertTrue(exact_match("Graph search, with provenance.", "graph search with provenance"))

    def test_judge_parses_json_fenced_response(self):
        class FakeJudge:
            def invoke(self, messages):
                return SimpleNamespace(
                    content='```json\n{"correct": true, "score": 1, "reason": "Equivalent."}\n```'
                )

        result = judge_answer("Question", "Reference", "Actual", FakeJudge())
        self.assertTrue(result["correct"])
        self.assertEqual(result["score"], 1.0)

    def test_question_set_has_ten_examples_per_type(self):
        questions = runner.load_questions()
        counts = {}
        for question in questions:
            counts[question["type"]] = counts.get(question["type"], 0) + 1
        self.assertEqual(len(questions), 50)
        self.assertEqual(set(counts.values()), {10})

    def test_every_out_of_scope_question_triggers_refusal_guard(self):
        from src.guardrails.topic_guard import check_question

        out_of_scope = [item for item in runner.load_questions() if item["type"] == "out_of_scope"]
        unblocked = [item["question"] for item in out_of_scope if not check_question(item["question"]).blocked]
        self.assertEqual(unblocked, [])

    def test_runner_scores_both_systems_and_writes_report(self):
        question = {
            "id": "test-1",
            "type": "single_hop",
            "hop_count": 1,
            "question": "What database stores vectors?",
            "expected_answer": "PostgreSQL",
        }
        hybrid_result = {"answer": "PostgreSQL", "latency_ms": 12.0}
        with (
            patch.object(runner, "_vector_only_answer", return_value=("PostgreSQL", 8.0)),
            patch.object(runner, "retrieve_and_answer", return_value=hybrid_result),
        ):
            report = runner.run_benchmark([question], llm=object(), judge=False)

        self.assertEqual(len(report["results"]), 2)
        self.assertEqual(report["summary"]["vector"]["exact_match_accuracy"], 1.0)
        self.assertEqual(report["summary"]["hybrid"]["p95_latency_ms"], 12.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "benchmark.json"
            markdown_path = Path(temp_dir) / "benchmark.md"
            runner.write_reports(report, json_path, markdown_path)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["question_count"], 1)
            self.assertIn("Accuracy by Hop Count", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()