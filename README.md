# Hybrid Knowledge Graph RAG System

A production-grade, hybrid Retrieval-Augmented Generation (RAG) system combining **dense vector similarity search** (PostgreSQL + pgvector with HNSW index) and **graph traversal** (Neo4j Desktop) to answer both factual similarity and complex multi-hop relationship queries with full citation provenance.

---

## Architecture Overview

```
DOCUMENTS (PDFs, Reports, Papers)
    │
    ▼
[ Ingestion Pipeline ] ──► SHA-256 Idempotency Check
    │
    ├─► Chunker (Page provenance, word windows, char offsets)
    │     │
    │     ▼
    │   Embedder (Ollama nomic-embed-text: 768-dim)
    │     │
    │     ▼
    │   PostgreSQL + pgvector (HNSW cosine index)
    │
    └─► Entity & Relation Extractor (LLM JSON schema)
          │
          ▼
        Entity Resolver (Canonical normalization, alias merging)
          │
          ▼
        Neo4j Knowledge Graph (Nodes, directed edges, source chunk provenance)
```

---

## Key Features

- **Dense Vector Search (PostgreSQL + `pgvector`):**
  - Chunks indexed using HNSW (`m=16`, `ef_construction=128`) for sub-millisecond approximate nearest-neighbor search.
  - Page numbers and exact character offset boundaries recorded per chunk.
- **Knowledge Graph (Neo4j):**
  - Domain ontology across 8 entity types (`Person`, `Organization`, `Technology`, `Project`, etc.) and 12 relation types (`USES`, `DEVELOPS`, `MANAGES`, `PART_OF`, etc.).
  - Entity resolution with canonical naming, alias merging (e.g., `Postgres` $\rightarrow$ `PostgreSQL`), and duplicate prevention.
  - Complete edge provenance: each graph relation links back to its exact source `chunk_id` and document.
- **Idempotent Ingestion:**
  - Documents hashed via SHA-256. Re-uploading an existing document skips redundant embedding and LLM extraction.
- **Interactive Streamlit UI:**
  - Multi-document upload with real-time status.
  - Persistent document library with per-document deletion and cascade cleanups.
  - Interactive **Knowledge Graph Explorer** tab: view graph statistics, entity type distributions, and inspect 1-hop / 2-hop neighborhoods with provenance citations.

---

## Tech Stack & Prerequisites

| Component | Tool / Model | Description |
|-----------|--------------|-------------|
| **Language** | Python 3.11+ | Backend logic and pipelines |
| **Vector DB** | PostgreSQL 16 + `pgvector` | HNSW cosine similarity storage |
| **Graph DB** | Neo4j Desktop (v5+ / 2026.x) | Graph storage and multi-hop Cypher queries |
| **Embeddings** | Ollama `nomic-embed-text` | 768-dimensional dense embeddings |
| **Chat LLM** | Ollama `llama3.2:1b` | Local generation (Claude & OpenAI API supported) |
| **Frontend** | Streamlit | Web application and graph inspector |

---

## Getting Started

### 1. Clone & Set Up Virtual Environment

```powershell
git clone <repo-url>
cd ai-engineering-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)

Copy the example configuration:
```powershell
cp .env.example .env
```

Edit `.env` to match your local setup:
```env
# PostgreSQL + pgvector
PG_HOST=localhost
PG_PORT=5433
PG_USER=rag_user
PG_PASSWORD=rag_password
PG_DATABASE=rag_db

# Neo4j Desktop (bolt direct connection)
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Ollama models
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2:1b
OLLAMA_EMBED_MODEL=nomic-embed-text
EMBEDDING_DIM=768
```

### 3. Start Local Services

- **Ollama:**
  ```powershell
  ollama pull llama3.2:1b
  ollama pull nomic-embed-text
  ```
- **PostgreSQL with pgvector:**
  ```powershell
  docker run -d --name rag-pgvector -p 5433:5432 -e POSTGRES_USER=rag_user -e POSTGRES_PASSWORD=rag_password -e POSTGRES_DB=rag_db pgvector/pgvector:pg16
  ```
- **Neo4j Desktop:**
  Open Neo4j Desktop, start your local DBMS, and verify it is listening on `bolt://127.0.0.1:7687`.

---

## Verification & Tests

Run the smoke test to verify all connections:
```powershell
python smoke_test.py
```

Run Phase 2 (Ingestion & pgvector search) end-to-end test:
```powershell
python test_phase2.py
```

Run Phase 3 (Entity extraction & Neo4j graph) end-to-end test:
```powershell
python test_phase3.py
```

---

## Launch the Application

```powershell
streamlit run main.py
```

Open your browser at `http://localhost:8501`.

---

## Project Roadmap

- [x] **Phase 1 — Infrastructure:** PostgreSQL + pgvector, Neo4j, Ollama, schema initialization, and smoke test.
- [x] **Phase 2 — Ingestion & Vector Search:** Rich chunking with page provenance, SHA-256 idempotency, batched embedding, and pgvector HNSW search.
- [x] **Phase 3 — Knowledge Graph Construction:** Domain ontology, LLM entity & relation extraction, canonical resolution, Neo4j graph writing with chunk provenance, and graph explorer UI.
- [ ] **Phase 4 — Query Router & Hybrid Retrieval:** Query intent classifier (`vector`, `graph`, `both`), path-to-prose conversion, and fused context generation.
- [ ] **Phase 5 — Evaluation & Benchmarking:** Automated multi-hop test set evaluation (RAGAS / TruLens) comparing pure vector vs. hybrid graph RAG.
- [ ] **Phase 6 — Guardrails & Safety:** Citation verification and anti-hallucination validation.