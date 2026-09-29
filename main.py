from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from math import sqrt

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from pypdf import PdfReader


EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:1b"
CHUNK_WORDS = 180
CHUNK_OVERLAP = 35
TOP_K = 5


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    text: str


def extract_chunks(pdf_bytes: bytes) -> list[TextChunk]:
    reader = PdfReader(BytesIO(pdf_bytes))
    chunks: list[TextChunk] = []
    step = CHUNK_WORDS - CHUNK_OVERLAP

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        words = page_text.split()
        for start in range(0, len(words), step):
            text = " ".join(words[start : start + CHUNK_WORDS])
            if text:
                chunks.append(TextChunk(page_number=page_number, text=text))

    return chunks


def embed_chunks(embeddings_model: OllamaEmbeddings, chunks: list[TextChunk]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), 100):
        batch = chunks[start : start + 100]
        embeddings.extend(embeddings_model.embed_documents([chunk.text for chunk in batch]))
    return embeddings


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot_product / (left_norm * right_norm)


def answer_question(
    chat_model: ChatOllama,
    question: str,
    chunks: list[TextChunk],
    chunk_embeddings: list[list[float]],
) -> tuple[str, list[TextChunk]]:
    question_embedding = OllamaEmbeddings(model=EMBEDDING_MODEL).embed_query(question)
    ranked = sorted(
        range(len(chunks)),
        key=lambda index: cosine_similarity(question_embedding, chunk_embeddings[index]),
        reverse=True,
    )
    sources = [chunks[index] for index in ranked[:TOP_K]]
    context = "\n\n".join(
        f"[Page {chunk.page_number}] {chunk.text}" for chunk in sources
    )

    completion = chat_model.invoke(
        [
            SystemMessage(
                content=(
                    "Answer the user's question using only the supplied PDF excerpts. "
                    "If the excerpts do not contain the answer, say so. Cite supporting "
                    "pages using the format [Page N]."
                )
            ),
            HumanMessage(content=f"PDF excerpts:\n{context}\n\nQuestion: {question}"),
        ]
    )
    answer = str(completion.content) or "I couldn't generate an answer."
    source_pages = dict.fromkeys(chunk.page_number for chunk in sources)
    citations = ", ".join(f"[Page {page_number}]" for page_number in source_pages)
    answer = f"{answer}\n\nSources: {citations}"
    return answer, sources


st.set_page_config(page_title="Ask your PDF")
st.title("Ask your PDF")
st.write("Upload a PDF, then ask questions grounded in its contents.")
st.sidebar.caption(f"Local models: {CHAT_MODEL} and {EMBEDDING_MODEL}")

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_digest = sha256(pdf_bytes).hexdigest()

    if st.button("Index PDF", type="primary"):
        try:
            chunks = extract_chunks(pdf_bytes)
            if not chunks:
                st.error("No readable text was found. This app needs a text-based PDF.")
            else:
                embeddings = embed_chunks(OllamaEmbeddings(model=EMBEDDING_MODEL), chunks)
                st.session_state["pdf_index"] = {
                    "digest": current_digest,
                    "chunks": chunks,
                    "embeddings": embeddings,
                }
                st.success(f"Indexed {len(chunks)} text chunks. Ask away.")
        except Exception as error:
            st.error(
                f"Could not index this PDF: {error}. Check that Ollama is running "
                f"and that {EMBEDDING_MODEL} is installed."
            )

    index = st.session_state.get("pdf_index")
    if index and index["digest"] == current_digest:
        question = st.text_input("Your question")
        if st.button("Get answer", disabled=not question.strip()):
            try:
                answer, sources = answer_question(
                    ChatOllama(model=CHAT_MODEL, temperature=0),
                    question.strip(),
                    index["chunks"],
                    index["embeddings"],
                )
                st.subheader("Answer")
                st.write(answer)
                with st.expander("Retrieved passages"):
                    for chunk in sources:
                        st.markdown(f"**Page {chunk.page_number}**")
                        st.write(chunk.text)
            except Exception as error:
                st.error(
                    f"Could not answer this question: {error}. Check that Ollama is "
                    f"running and that {CHAT_MODEL} is installed."
                )
    else:
        st.info("Index this PDF to start asking questions.")