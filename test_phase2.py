"""
test_phase2.py — Test Phase 2 ingestion pipeline end-to-end
"""
import io
from pypdf import PdfWriter
from src.ingestion.chunker import extract_chunks_from_pdf, extract_chunks_from_text
from src.ingestion.embedder import get_embedder
from src.ingestion.ingest import ingest_pdf, list_ingested_documents, delete_document
from src.db.postgres import vector_search


def create_dummy_pdf(text_pages: list[str]) -> bytes:
    """Create a minimal synthetic PDF in memory with reportlab or simple canvas if available, or pypdf."""
    # We can create a simple PDF with reportlab or fpdf if present, or let's check
    pass


def main():
    print("=== Testing Phase 2 Chunker ===")
    sample_text = (
        "Retrieval-Augmented Generation (RAG) optimizes the output of a large language model. "
        "It references an authoritative knowledge base outside of its training data sources before generating a response. "
        "Knowledge graphs organize data from multiple sources, capture information about entities of interest, "
        "and establish relationships among them. In this architecture, Neo4j handles multi-hop graph traversals "
        "while PostgreSQL with pgvector handles dense vector similarity queries."
    )
    chunks = extract_chunks_from_text(sample_text, filename="sample.txt", chunk_words=20, chunk_overlap=5)
    print(f"Extracted {len(chunks)} chunks from text:")
    for c in chunks:
        print(f"  [Chunk {c.chunk_index}] words={c.metadata.get('word_count')} span=({c.char_start}..{c.char_end}): {c.text[:50]}...")

    print("\n=== Testing Phase 2 Embedder ===")
    embedder = get_embedder()
    test_texts = [c.text for c in chunks[:2]]
    embeddings = embedder.embed_texts(test_texts)
    print(f"Generated {len(embeddings)} embeddings. Dimension: {len(embeddings[0])}")
    assert len(embeddings[0]) == 768, f"Expected 768, got {len(embeddings[0])}"

    print("\n=== Testing PDF Ingestion & Idempotency ===")
    # Let's create a real minimal PDF with pypdf or reportlab
    from pypdf import PdfWriter
    writer = PdfWriter()
    # pypdf doesn't easily create blank text pages without a base, let's use reportlab or write a minimal valid PDF
    # Minimal 1-page PDF stream:
    pdf_bytes = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 145 >> stream
BT
/F1 12 Tf
72 712 Td
(Knowledge graph RAG combines vector search with graph traversal.) Tj
0 -20 Td
(Neo4j is used for multi-hop graph queries.) Tj
ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000440 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
514
%%EOF"""

    res1 = ingest_pdf(pdf_bytes, filename="knowledge_graph_intro.pdf", force=False)
    print(f"First ingestion result: status={res1.status}, doc_id={res1.doc_id}, chunks={res1.chunks_count}, time={res1.elapsed_seconds}s")
    assert res1.status in ("ingested", "skipped"), f"Unexpected status: {res1.status}"

    res2 = ingest_pdf(pdf_bytes, filename="knowledge_graph_intro.pdf", force=False)
    print(f"Second ingestion result: status={res2.status}, doc_id={res2.doc_id}, chunks={res2.chunks_count}, message='{res2.message}'")
    assert res2.status == "skipped", f"Expected skipped, got {res2.status}"

    print("\n=== Testing Vector Search in pgvector ===")
    query_emb = embedder.embed_query("What does Neo4j do in multi-hop graph queries?")
    results = vector_search(query_emb, top_k=2)
    print(f"Vector search returned {len(results)} matches:")
    for r in results:
        print(f"  [doc_id={r['doc_id']} chunk={r['chunk_index']} score={r['score']:.4f}]: {r['text']}")

    docs = list_ingested_documents()
    print(f"\nIngested documents in PostgreSQL: {len(docs)}")
    for d in docs:
        print(f"  ID: {d['id']}, File: {d['filename']}, Chunks: {d['chunk_count']}, Ingested: {d['ingested_at']}")

    print("\n>>> Phase 2 pipeline test PASSED successfully! <<<")


if __name__ == "__main__":
    main()
