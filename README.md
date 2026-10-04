# Hybrid Knowledge Graph RAG System | Benchmark pending: 50 labeled questions; results not measured

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

Run Phase 4 retrieval unit tests (no database or model services required):
```powershell
& .\.venv\Scripts\python.exe -m unittest test_phase4
```

Run Phase 5 citation validation and generation tests:
```powershell
& .\.venv\Scripts\python.exe -m unittest test_phase5
```

Run both suites together:
```powershell
& .\.venv\Scripts\python.exe -m unittest test_phase4 test_phase5
```

Run the offline evaluation tests:
```powershell
& .\.venv\Scripts\python.exe -m unittest test_eval
```

### Benchmark Suite

The benchmark compares the existing hybrid retriever with a vector-only baseline on the same 50 labeled questions: 10 each for single-hop, two-hop, three-hop, aggregation, and out-of-scope refusal. Labels are in `eval_questions.jsonl`; exact-match scoring is always recorded, and non-exact answers are sent to the configured LLM judge.

The questions describe facts from this README. The application ingests PDFs rather than Markdown, so first create and ingest a PDF containing the documented facts, then start PostgreSQL, Neo4j, and Ollama (or configure a supported chat provider). Without that corpus, in-scope questions cannot produce a meaningful retrieval comparison. Run:

```powershell
& .\.venv\Scripts\python.exe -m src.eval.runner
```

The run writes detailed results to `reports/benchmark.json` and summary tables to `reports/benchmark.md`, including accuracy by question type and hop count and p95 latency. Use `--no-judge` for an exact-match-only run. API cost per query is currently reported as unavailable because token usage and provider pricing are not instrumented. Pass `--ingestion-cost-usd` to include a separately measured ingestion cost in the Markdown table; do not treat missing values as zero.

No live benchmark has been run yet. The README's first-line status remains pending until measured results are available; no accuracy, latency, or cost figures are inferred from ordinary retrieval logs.

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
- [x] **Phase 4 — Query Router & Hybrid Retrieval:** Confidence-aware intent routing, entity-linked graph retrieval, path-to-prose conversion, labeled context assembly, and JSONL query telemetry.
- [x] **Phase 5 — Citation Validator & Answer Generation:** Require every factual sentence to cite a retrieved chunk ID, validate citations and regenerate up to two times, with Ollama-first optional Claude/OpenAI fallback.
- [x] **Phase 6 — Guardrails & Safety:** Pre-retrieval topic guard, post-generation groundedness checker, and PII scrubber.

Phase 4 appends one JSON record per completed question to `logs/retrieval.jsonl` with its route, confidence, latency, and answer. Set `ROUTE_CONFIDENCE_THRESHOLD` (default `0.65`) in `.env` to control when uncertain classifications fall back to both retrieval paths.

Phase 5 records citation validity, generation attempts, retry count, retry rate, cited and invalid chunk IDs, uncited claim count, and generation provider in the same JSONL log. Citations use `[chunk_id: N]` and are accepted only when `N` belongs to the retrieved evidence set. If generated text fails citation checks twice, the app shows up to three sentences copied from retrieved evidence with verified citations. If there is no retrieved evidence, it abstains rather than presenting an unverified answer.

Phase 6 adds three guardrail layers:

1. **Pre-retrieval topic guard** — Regex + keyword blocklist checks every question for jailbreak/prompt-injection attempts, harmful content requests, PII exfiltration, and clearly off-topic prompts. A blocked question is refused immediately without any LLM or database call.
2. **Post-generation groundedness checker** — Lexical token-overlap check verifies that each cited sentence's key content words appear in the body of the chunk it references. Sentences that cite a valid chunk ID but assert facts absent from that chunk are flagged as "ungrounded" (citation laundering). Set `GROUNDEDNESS_MIN_OVERLAP` (default `0.25`) to tune strictness.
3. **PII scrubber** — Regex patterns redact emails, phone numbers, SSNs, credit card numbers, IPv4 addresses, passport numbers, and UK NI numbers from the final answer text before display or logging.

Set `GUARDRAILS_ENABLED=false` in `.env` to bypass all Phase 6 checks (useful during integration testing). All guardrail signals are appended to `logs/retrieval.jsonl`.

Ollama is used first for answer generation. Set `LLM_FALLBACK_ENABLED=true` and `LLM_FALLBACK_PROVIDER=anthropic` or `openai` to use that cloud provider when Ollama invocation fails. Citation-format failures instead trigger regeneration and then the verified-source fallback.

### Manual Retrieval Check

1. Start PostgreSQL, Neo4j, and Ollama. PgAdmin is optional; it is a database client, not the PostgreSQL server.
2. Launch the app with `streamlit run main.py`, then ingest a document that contains the facts you want to ask about. Graph extraction should be enabled for relationship questions.
3. Ask a definition question and a relationship question from that document. Check the route, answer citations, and retrieved evidence in the UI. Questions about facts absent from the ingested documents should not be expected to produce an answer.
4. Try a jailbreak or off-topic question to verify the guardrail block banner appears without any retrieval occurring.
5. Inspect recent routing, citation, and guardrail diagnostics:

```powershell
Get-Content .\logs\retrieval.jsonl -Tail 5 |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Select-Object question, route, confidence, citation_valid,
    guardrails_blocked, guardrails_block_category,
    groundedness_score, grounded, pii_scrubbed |
  Format-Table -AutoSize
```

A route confidence of `0.00` means the router response could not be parsed and the request used both retrieval paths as a fallback. High latency or repeated citation retries can indicate the local model is struggling with the requested output format; check the Streamlit terminal and the log fields above.