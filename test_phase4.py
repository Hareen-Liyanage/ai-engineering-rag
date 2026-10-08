"""Offline unit tests for Phase 4 query routing and context assembly."""
import unittest

from src.retrieval.assembler import assemble_context
from src.retrieval.graph_search import GRAPH_QUERY_TEMPLATES, triples_to_prose
from src.retrieval.router import RetrievalRoute, _parse_decision, classify_question


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, messages):
        return _Response(self.content)


class _FailingLLM:
    def invoke(self, messages):
        raise RuntimeError("model unavailable")


class Phase4Tests(unittest.TestCase):
    def test_confident_relationship_question_routes_to_graph(self):
        llm = _FakeLLM('{"route": "graph", "confidence": 0.92}')
        decision = classify_question("Who manages Project Gemini?", llm=llm)
        self.assertEqual(decision.route, RetrievalRoute.GRAPH)
        self.assertFalse(decision.fallback)

    def test_confident_definition_question_routes_to_vector(self):
        llm = _FakeLLM('{"route": "vector", "confidence": 0.91}')
        decision = classify_question("What does RAG mean?", llm=llm)
        self.assertEqual(decision.route, RetrievalRoute.VECTOR)
        self.assertFalse(decision.fallback)

    def test_low_confidence_routes_to_both(self):
        llm = _FakeLLM('{"route": "vector", "confidence": 0.2}')
        decision = classify_question("What does RAG mean?", llm=llm)
        self.assertEqual(decision.route, RetrievalRoute.BOTH)
        self.assertTrue(decision.fallback)

    def test_malformed_router_output_falls_back(self):
        route, confidence = _parse_decision("not json")
        self.assertEqual(route, RetrievalRoute.BOTH)
        self.assertEqual(confidence, 0.0)

    def test_router_model_error_falls_back_to_both(self):
        decision = classify_question("What does RAG mean?", llm=_FailingLLM())
        self.assertEqual(decision.route, RetrievalRoute.BOTH)
        self.assertTrue(decision.fallback)

    def test_graph_templates_parameterize_inputs(self):
        self.assertIn("$node_ids", GRAPH_QUERY_TEMPLATES["relationships"])
        self.assertIn("$doc_ids", GRAPH_QUERY_TEMPLATES["relationships"])
        self.assertIn("$limit", GRAPH_QUERY_TEMPLATES["relationships"])
        self.assertIn("$node_ids", GRAPH_QUERY_TEMPLATES["manages"])

    def test_multi_hop_template_is_bounded_and_parameterized(self):
        template = GRAPH_QUERY_TEMPLATES["multi_hop"]
        self.assertIn("[*1..$max_hops]", template)
        self.assertIn("$node_ids", template)
        self.assertIn("$doc_ids", template)
        self.assertIn("$limit", template)
        self.assertIn("relationships(path)", template)

    def test_graph_triples_become_deduplicated_prose(self):
        triple = {
            "source": "Alice",
            "source_type": "Person",
            "relation": "MANAGES",
            "target": "Project X",
            "target_type": "Project",
            "source_chunk_id": 12,
            "doc_id": 4,
        }
        prose = triples_to_prose([triple, triple])
        self.assertEqual(len(prose), 1)
        self.assertEqual(prose[0]["text"], "Alice (Person) manages Project X (Project).")

    def test_context_labels_and_deduplicates_evidence(self):
        vector_hit = {"id": 12, "doc_id": 4, "chunk_index": 2, "page_number": 3, "text": "A passage."}
        graph_hit = {"text": "Alice manages Project X.", "doc_id": 4, "source_chunk_id": 12}
        context = assemble_context(
            [vector_hit, vector_hit],
            [graph_hit, graph_hit],
            [{"id": 4, "filename": "report.pdf"}],
        )
        self.assertEqual(context.count("A passage."), 1)
        self.assertEqual(context.count("Alice manages Project X."), 1)
        self.assertIn("[VECTOR | report.pdf | Page 3 | chunk_id: 12 | Chunk index: 2]", context)
        self.assertIn("[GRAPH | report.pdf | chunk_id: 12]", context)


if __name__ == "__main__":
    unittest.main()