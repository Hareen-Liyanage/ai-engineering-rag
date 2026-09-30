"""Build a deduplicated, source-labeled prompt from retrieval results."""
from __future__ import annotations


def assemble_context(
    vector_results: list[dict],
    graph_results: list[dict],
    documents: list[dict] | None = None,
) -> str:
    """Format vector passages and graph statements with readable citations."""
    filenames = {doc["id"]: doc["filename"] for doc in documents or []}
    sections: list[str] = []
    seen_vector_chunks: set[int] = set()
    seen_graph_facts: set[str] = set()

    for result in vector_results:
        chunk_id = result.get("id")
        if chunk_id in seen_vector_chunks:
            continue
        if chunk_id is not None:
            seen_vector_chunks.add(chunk_id)
        filename = filenames.get(result.get("doc_id"), f"Doc #{result.get('doc_id')}")
        sections.append(
            f"[VECTOR | {filename} | Page {result.get('page_number')} | "
            f"chunk_id: {chunk_id} | Chunk index: {result.get('chunk_index')}]\n"
            f"{result.get('text', '')}"
        )

    for result in graph_results:
        text = str(result.get("text", "")).strip()
        chunk_id = result.get("source_chunk_id")
        if not text or chunk_id is None or text.casefold() in seen_graph_facts:
            continue
        seen_graph_facts.add(text.casefold())
        filename = filenames.get(result.get("doc_id"), f"Doc #{result.get('doc_id')}")
        citation = f"{filename} | chunk_id: {chunk_id}"
        sections.append(f"[GRAPH | {citation}]\n{text}")

    return "\n\n---\n\n".join(sections)


def build_messages(question: str, context: str) -> list:
    """Create the grounded-answer chat messages for the configured LLM."""
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(
            content=(
                "You are an expert AI knowledge assistant. Answer strictly using the provided evidence. "
                "Every factual claim must be its own sentence and end with a citation in the exact format "
                "[chunk_id: N], using only chunk IDs shown in the evidence. Never invent a chunk ID. "
                "If the evidence does not support an answer, state that no relevant evidence was found."
            )
        ),
        HumanMessage(content=f"Evidence:\n{context or '(No relevant evidence found.)'}\n\nQuestion: {question}"),
    ]