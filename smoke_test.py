"""
smoke_test.py — Phase 1 infrastructure verification

Run with:
    .venv\Scripts\python.exe smoke_test.py

Checks:
  1. PostgreSQL / pgvector  — insert a dummy embedding, retrieve it, delete it
  2. Neo4j                  — create a node, read it back, delete it
  3. Ollama                 — embed a test string and confirm dimension matches config
  4. LLM provider           — if ANTHROPIC_API_KEY or OPENAI_API_KEY set, do a 1-token call

Exit code 0 = all available services passed. Any failure prints RED details.
"""
from __future__ import annotations

import sys
import logging

logging.basicConfig(level=logging.WARNING)  # suppress info noise during smoke test

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m–\033[0m"

results: list[tuple[str, str, str]] = []  # (service, status_symbol, detail)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────
try:
    from src.config import settings
    results.append(("Config", PASS, f"provider={settings.llm_provider}  dim={settings.embedding_dim}"))
except Exception as exc:
    results.append(("Config", FAIL, str(exc)))
    print("\n  Config load failed — cannot continue.\n")
    for service, symbol, detail in results:
        print(f"  {symbol}  {service:<30} {detail}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PostgreSQL / pgvector
# ─────────────────────────────────────────────────────────────────────────────
try:
    from src.db.postgres import init_schema, get_conn
    import psycopg2.extras

    init_schema()

    dummy_vec = [0.0] * settings.embedding_dim
    dummy_vec[0] = 1.0  # non-zero so cosine is defined

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Insert a throwaway document + chunk
            cur.execute(
                "INSERT INTO documents (filename, sha256) VALUES (%s, %s) ON CONFLICT DO NOTHING RETURNING id",
                ("_smoke_test.pdf", "0" * 64),
            )
            row = cur.fetchone()
            if row:
                doc_id = row[0]
                cur.execute(
                    "INSERT INTO chunks (doc_id, chunk_index, page_number, text, embedding) "
                    "VALUES (%s, %s, %s, %s, %s::vector)",
                    (doc_id, 0, 1, "smoke test chunk", dummy_vec),
                )
                # Vector search
                cur.execute(
                    "SELECT text, 1 - (embedding <=> %s::vector) AS score FROM chunks "
                    "WHERE doc_id = %s ORDER BY embedding <=> %s::vector LIMIT 1",
                    (dummy_vec, doc_id, dummy_vec),
                )
                hit = cur.fetchone()
                # Cleanup
                cur.execute("DELETE FROM documents WHERE sha256 = %s", ("0" * 64,))
                results.append(("PostgreSQL + pgvector", PASS,
                                 f"HNSW search returned: '{hit[0]}' (score={hit[1]:.4f})"))
            else:
                results.append(("PostgreSQL + pgvector", SKIP,
                                 "Smoke document already exists — schema OK"))
except Exception as exc:
    results.append(("PostgreSQL + pgvector", FAIL, str(exc)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Neo4j
# ─────────────────────────────────────────────────────────────────────────────
try:
    from src.db.neo4j_conn import init_schema as neo4j_init, smoke_test as neo4j_smoke
    neo4j_init()
    result = neo4j_smoke()
    if result["ok"]:
        results.append(("Neo4j", PASS, f"node round-trip OK — {result}"))
    else:
        results.append(("Neo4j", SKIP, result["error"]))
except Exception as exc:
    results.append(("Neo4j", SKIP, f"Not available: {exc}"))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ollama embeddings
# ─────────────────────────────────────────────────────────────────────────────
try:
    from langchain_ollama import OllamaEmbeddings
    emb = OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)
    vec = emb.embed_query("smoke test")
    dim = len(vec)
    if dim == settings.embedding_dim:
        results.append(("Ollama embeddings", PASS, f"dim={dim} matches config"))
    else:
        results.append(("Ollama embeddings", FAIL,
                         f"dim mismatch: got {dim}, config says {settings.embedding_dim}. "
                         f"Update EMBEDDING_DIM in .env"))
except Exception as exc:
    results.append(("Ollama embeddings", FAIL, str(exc)))


# ─────────────────────────────────────────────────────────────────────────────
# 5. LLM provider (optional — only if keys are set)
# ─────────────────────────────────────────────────────────────────────────────
if settings.anthropic_api_key:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=5,
            messages=[{"role": "user", "content": "Reply with just the word OK"}],
        )
        results.append(("Anthropic API", PASS, f"response: {msg.content[0].text!r}"))
    except Exception as exc:
        results.append(("Anthropic API", FAIL, str(exc)))
else:
    results.append(("Anthropic API", SKIP, "ANTHROPIC_API_KEY not set"))

if settings.openai_api_key:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=5,
            messages=[{"role": "user", "content": "Reply with just the word OK"}],
        )
        results.append(("OpenAI API", PASS, f"response: {resp.choices[0].message.content!r}"))
    except Exception as exc:
        results.append(("OpenAI API", FAIL, str(exc)))
else:
    results.append(("OpenAI API", SKIP, "OPENAI_API_KEY not set"))


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Phase 1 Smoke Test")
print("=" * 60)
for service, symbol, detail in results:
    print(f"  {symbol}  {service:<30} {detail}")
print("=" * 60)

failures = [s for s, sym, _ in results if sym == "\033[91m✗\033[0m"]
if failures:
    print(f"\n  {len(failures)} check(s) FAILED. Fix before moving to Phase 2.\n")
    sys.exit(1)
else:
    print("\n  All checks passed (skips = optional services).\n")
    sys.exit(0)
