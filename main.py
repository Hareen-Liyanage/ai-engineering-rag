from __future__ import annotations

import streamlit as st

from src.config import settings
from src.db import postgres, neo4j_conn
from src.ingestion.ingest import (
    ingest_pdf,
    list_ingested_documents,
    delete_document,
)
from src.graph.writer import get_graph_stats, query_entity_neighborhood
from src.retrieval.router import retrieve_and_answer

# Ensure schemas are ready
postgres.init_schema()
neo4j_conn.init_schema()

st.set_page_config(
    page_title="Knowledge Graph RAG Assistant",
    page_icon="🕸️",
    layout="wide",
)

st.title("🕸️ Knowledge Graph RAG — Hybrid Retrieval Assistant")
st.caption(
    f"Hybrid Backend: PostgreSQL + pgvector (HNSW) & Neo4j Desktop | "
    f"Ollama: {settings.ollama_chat_model} & {settings.ollama_embed_model}"
)

# ── Sidebar: Document Library & Graph Health ─────────────────────────────────
with st.sidebar:
    st.header("🗂️ Document Library")
    st.write("Documents stored persistently in PostgreSQL:")

    docs = list_ingested_documents()
    if not docs:
        st.info("No documents ingested yet. Upload PDFs to get started.")
    else:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{doc['filename']}**")
                st.caption(f"ID: {doc['id']} | {doc['chunk_count']} chunks | {str(doc['ingested_at'])[:10]}")
            with col2:
                if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document & graph nodes"):
                    delete_document(doc["id"])
                    st.rerun()

    st.divider()
    st.header("📊 Knowledge Graph Stats")
    g_stats = get_graph_stats()
    if "error" in g_stats:
        st.warning(f"Neo4j: {g_stats['error']}")
    else:
        st.metric("Graph Entities", g_stats.get("node_count", 0))
        st.metric("Graph Relationships", g_stats.get("edge_count", 0))

    st.divider()
    st.subheader("⚙️ Settings")
    st.text(f"Vector Dim: {settings.embedding_dim}")
    st.text(f"Top-K: {settings.top_k}")
    st.text(f"Chunk Words: {settings.chunk_words}")

# ── Main Content Area ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📤 Upload & Ingest Documents", "🕸️ Knowledge Graph Explorer"])

with tab2:
    st.subheader("Upload PDFs to Knowledge Base")
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col_force, col_graph = st.columns(2)
    with col_force:
        force_reingest = st.checkbox("Force re-ingest (overwrite existing chunks)", value=False)
    with col_graph:
        build_kg = st.checkbox("Extract Knowledge Graph (Neo4j Entities & Triples)", value=True)

    if uploaded_files and st.button("Ingest Documents", type="primary"):
        progress_bar = st.progress(0)
        status_box = st.container()

        for idx, file in enumerate(uploaded_files):
            pdf_bytes = file.getvalue()
            with status_box:
                st.write(f"Processing **{file.name}**...")
                res = ingest_pdf(
                    pdf_bytes,
                    filename=file.name,
                    force=force_reingest,
                    build_graph=build_kg,
                )
                if res.status == "ingested":
                    st.success(f"✅ {file.name}: {res.message}")
                elif res.status == "skipped":
                    st.info(f"⏭️ {file.name}: {res.message}")
                else:
                    st.error(f"❌ {file.name}: {res.message}")

            progress_bar.progress((idx + 1) / len(uploaded_files))

        st.success("All uploads processed!")
        st.rerun()

with tab3:
    st.subheader("🕸️ Neo4j Knowledge Graph Visualizer & Inspector")
    stats = get_graph_stats()

    if "error" in stats:
        st.error(f"Cannot connect to Neo4j: {stats['error']}")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entity Nodes", stats.get("node_count", 0))
        with col2:
            st.metric("Total Relations", stats.get("edge_count", 0))
        with col3:
            st.metric("Ontology Types Present", len(stats.get("type_breakdown", {})))

        st.markdown("### Entity Types Breakdown")
        breakdown = stats.get("type_breakdown", {})
        if breakdown:
            st.bar_chart(breakdown)

        st.divider()
        st.markdown("### Explore Entity Neighborhood")
        top_ents = [e["name"] for e in stats.get("top_entities", [])]
        search_target = st.text_input(
            "Search an entity name (or pick a top connected entity):",
            value=top_ents[0] if top_ents else "",
        )

        if search_target.strip():
            triples = query_entity_neighborhood(search_target.strip())
            if not triples:
                st.info(f"No direct graph connections found for '{search_target}'.")
            else:
                st.write(f"Found **{len(triples)}** connections for **{search_target}**:")
                table_data = [
                    {
                        "Source Entity": t["source"],
                        "Source Type": t["source_type"],
                        "Relationship": f"-[ {t['relation']} ]->",
                        "Target Entity": t["target"],
                        "Target Type": t["target_type"],
                        "Provenance (Chunk ID)": t["source_chunk_id"],
                    }
                    for t in triples
                ]
                st.dataframe(table_data, use_container_width=True)

with tab1:
    docs = list_ingested_documents()
    if not docs:
        st.warning("Please upload and ingest at least one PDF in the 'Upload & Ingest' tab first.")
    else:
        doc_options = {d["id"]: f"{d['filename']} ({d['chunk_count']} chunks)" for d in docs}
        selected_doc_ids = st.multiselect(
            "Filter search to specific documents (leave empty to search all):",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
        )

        question = st.text_input(
            "Enter your question:",
            placeholder="e.g. What are the key findings, entities, or connections discussed?",
        )

        if st.button("Search & Answer", type="primary", disabled=not question.strip()):
            with st.spinner("Routing question and retrieving evidence..."):
                result = retrieve_and_answer(
                    question.strip(),
                    doc_ids=selected_doc_ids or None,
                    documents=docs,
                )

            decision = result["decision"]
            fallback_label = " (low-confidence fallback)" if decision.fallback else ""
            guardrails = result.get("guardrails")

            # ── Guardrail block banner ─────────────────────────────────────
            if guardrails and guardrails.blocked:
                st.error(
                    f"🛡️ **Request blocked by guardrails** "
                    f"({guardrails.topic_guard.category})\n\n"
                    f"{guardrails.topic_guard.reason}"
                )
            else:
                # Build guardrail detail string for caption
                guard_parts = []
                if guardrails and guardrails.groundedness is not None:
                    score_pct = round(guardrails.groundedness.mean_score * 100, 1)
                    grounded_label = "✅ grounded" if guardrails.grounded else "⚠️ ungrounded"
                    guard_parts.append(f"Groundedness: {score_pct}% ({grounded_label})")
                if guardrails and guardrails.pii_scrubbed:
                    guard_parts.append("🔒 PII scrubbed")
                guard_str = " | ".join(guard_parts)

                st.caption(
                    f"Route: {decision.route.value} | Confidence: {decision.confidence:.2f}"
                    f"{fallback_label} | {result['latency_ms']:.0f} ms | "
                    f"Citation retries: {result['generation'].retries}"
                    + (f" | {guard_str}" if guard_str else "")
                )
                st.subheader("💡 Answer")
                if result["generation"].extractive_fallback:
                    st.info("The generated answer failed citation checks. Showing verified source text instead.")
                if guardrails and not guardrails.grounded and guardrails.groundedness:
                    st.warning(
                        f"⚠️ Groundedness check flagged "
                        f"{len(guardrails.groundedness.ungrounded_sentences)} sentence(s) "
                        "as potentially not supported by the retrieved evidence."
                    )
                st.write(result["answer"])

            vector_matches = result["vector_results"]
            graph_matches = result["graph_results"]
            if not vector_matches and not graph_matches:
                st.warning("No relevant evidence found in the selected documents.")

            if vector_matches:
                st.subheader("🔍 Vector Evidence")
                for idx, match in enumerate(vector_matches, 1):
                    doc_name = next(
                        (doc["filename"] for doc in docs if doc["id"] == match["doc_id"]),
                        f"Doc #{match['doc_id']}",
                    )
                    score_pct = round(float(match["score"]) * 100, 1)
                    with st.expander(
                        f"#{idx} | {doc_name} (Page {match['page_number']}) — Similarity: {score_pct}%"
                    ):
                        st.write(match["text"])

            if graph_matches:
                st.subheader("🕸️ Graph Evidence")
                for idx, match in enumerate(graph_matches, 1):
                    st.write(f"{idx}. {match['text']}")
                    st.caption(
                        f"{match.get('source', 'graph').upper()} | "
                        f"Doc #{match.get('doc_id')} | Chunk {match.get('source_chunk_id')}"
                    )