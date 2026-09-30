"""
src/ingestion — Document parsing, chunking, embedding, and ingestion pipeline
"""
from src.ingestion.chunker import TextChunk, extract_chunks_from_pdf, extract_chunks_from_text
from src.ingestion.embedder import Embedder, get_embedder
from src.ingestion.ingest import ingest_pdf, list_ingested_documents, IngestionResult

__all__ = [
    "TextChunk",
    "extract_chunks_from_pdf",
    "extract_chunks_from_text",
    "Embedder",
    "get_embedder",
    "ingest_pdf",
    "list_ingested_documents",
    "IngestionResult",
]
