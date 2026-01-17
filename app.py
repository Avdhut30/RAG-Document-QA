import os
import shutil
import tempfile
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ----------------------------
# Config
# ----------------------------
APP_TITLE = "AI Document Q&A (RAG)"
VECTOR_DIR = "vectorstore"  # optional persistent storage

st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
st.title("📄 AI Document Q&A (RAG + Local LLM)")
st.success("🟢 Running in LOCAL MODE (Ollama • No API keys • 100% Free)")
st.caption("Upload PDFs → build a knowledge base → ask questions with grounded answers + citations.")

# ----------------------------
# Helpers
# ----------------------------
def load_pdfs_to_docs(pdf_files) -> List:
    """Load uploaded PDFs into LangChain Documents."""
    docs = []

    for f in pdf_files:
        # Streamlit uploader returns a BytesIO-like object; write to temp file for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            file_docs = loader.load()
        except Exception as e:
            # If PDF is encrypted or unsupported, skip gracefully
            st.warning(f"Skipped '{f.name}' (encrypted/unsupported). Error: {e}")
            file_docs = []

        # Add filename metadata
        for d in file_docs:
            d.metadata["source_file"] = f.name

        docs.extend(file_docs)

        # cleanup temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return docs


def chunk_docs(docs: List, chunk_size=1000, chunk_overlap=150) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def wipe_vector_dir():
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR, ignore_errors=True)


def build_faiss_from_chunks(chunks: List, save_to_disk: bool) -> FAISS:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = FAISS.from_documents(chunks, embeddings)

    if save_to_disk:
        os.makedirs(VECTOR_DIR, exist_ok=True)
        vectordb.save_local(VECTOR_DIR)

    return vectordb


def load_faiss_from_disk() -> FAISS:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)


def format_citations(docs: List) -> str:
    """Format citations from retrieved docs."""
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source_file", d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", None)
        page_str = f"p.{page + 1}" if isinstance(page, int) else "p.?"
        snippet = d.page_content.strip().replace("\n", " ")
        snippet = snippet[:180] + ("..." if len(snippet) > 180 else "")
        lines.append(f"[{i}] {src} ({page_str}) — {snippet}")
    return "\n".join(lines)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Answer ONLY using the provided context.\n"
            "Rules:\n"
            "1) If the answer is not in the context, say: \"I don't know based on the provided documents.\"\n"
            "2) Do NOT make up facts.\n"
            "3) Be concise and clear.\n"
            "4) Provide a short bullet list of key points.\n",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:",
        ),
    ]
)


def answer_question(vectordb: FAISS, question: str, k: int = 4) -> Tuple[str, List]:
    # Better retrieval: MMR helps diversity + relevance
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(10, k * 3), "lambda_mult": 0.5},
    )

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        [
            f"Source: {d.metadata.get('source_file','unknown')} | page: {d.metadata.get('page','?')}\n{d.page_content}"
            for d in retrieved_docs
        ]
    )

    llm = ChatOllama(model="llama3.2", temperature=0)
    chain = PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context})
    return result, retrieved_docs


# ----------------------------
# Session state init
# ----------------------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# ----------------------------
# UI (Sidebar)
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    chunk_size = st.slider("Chunk size", 300, 2000, 1000, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 10)
    top_k = st.slider("Top-k retrieval", 2, 10, 4, 1)

    st.divider()

    st.subheader("📄 Documents")
    pdf_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    save_index = st.checkbox("Save vector index to disk (optional)", value=False)
    load_from_disk = st.checkbox("Load existing index from disk (if available)", value=False)

    st.caption("Tip: For new PDFs, click **Build Knowledge Base** to rebuild the index.")

    st.divider()

    colA, colB = st.columns(2)
    build_btn = colA.button("📌 Build Knowledge Base", use_container_width=True)
    clear_btn = colB.button("🧹 Clear Index", use_container_width=True)

    st.divider()
    st.subheader("🧠 Models")
    st.write("LLM: **llama3.2**")
    st.write("Embeddings: **nomic-embed-text**")
    st.caption("Make sure Ollama is running and models are pulled.")

# ----------------------------
# Actions
# ----------------------------
if clear_btn:
    wipe_vector_dir()
    st.session_state.vectordb = None
    st.session_state.last_answer = ""
    st.session_state.last_sources = []
    st.success("Cleared vector index and reset session state.")

if build_btn:
    # Always rebuild from current uploaded PDFs (prevents stale answers)
    if load_from_disk and os.path.exists(VECTOR_DIR):
        try:
            st.session_state.vectordb = load_faiss_from_disk()
            st.session_state.last_answer = ""
            st.session_state.last_sources = []
            st.success("Loaded existing FAISS index from disk.")
        except Exception as e:
            st.error(f"Failed to load index from disk: {e}")
            st.session_state.vectordb = None
    else:
        if not pdf_files:
            st.error("Please upload at least one PDF.")
        else:
            # Reset old state to prevent “same output” confusion
            st.session_state.vectordb = None
            st.session_state.last_answer = ""
            st.session_state.last_sources = []

            if save_index:
                wipe_vector_dir()

            with st.spinner("Loading PDFs..."):
                docs = load_pdfs_to_docs(pdf_files)

            if not docs:
                st.error("No readable pages found. PDFs may be encrypted or empty.")
            else:
                with st.spinner("Chunking documents..."):
                    chunks = chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                with st.spinner("Building FAISS index..."):
                    st.session_state.vectordb = build_faiss_from_chunks(chunks, save_to_disk=save_index)

                st.success(f"✅ Knowledge base rebuilt from {len(pdf_files)} PDF(s). Pages: {len(docs)} | Chunks: {len(chunks)}")

# ----------------------------
# Main UI
# ----------------------------
st.divider()
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask a question")

    question = st.text_input("Type your question (e.g., 'What is the warranty period?')")

    ask_btn = st.button("Ask⚡", key="ask_btn")


    if ask_btn:
        if st.session_state.vectordb is None:
            st.warning("Upload PDFs and click **Build Knowledge Base** first.")
        elif not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Thinking with RAG (local)..."):
                answer, sources = answer_question(st.session_state.vectordb, question, k=top_k)
                st.session_state.last_answer = answer
                st.session_state.last_sources = sources

    if st.session_state.last_answer:
        st.markdown("### ✅ Answer")
        st.write(st.session_state.last_answer)

        st.markdown("### 📌 Citations")
        st.code(format_citations(st.session_state.last_sources))

        with st.expander("🔎 Show retrieved context (debug)"):
            for i, d in enumerate(st.session_state.last_sources, start=1):
                src = d.metadata.get("source_file", "unknown")
                page = d.metadata.get("page", None)
                page_str = f"{page + 1}" if isinstance(page, int) else "?"
                st.markdown(f"**[{i}] {src} | page {page_str}**")
                text = d.page_content.strip()
                st.write(text[:900] + ("..." if len(text) > 900 else ""))
                st.divider()

with col2:
    st.subheader("📦 Project Notes")
    st.markdown(
        "- Uses **RAG** (retrieval + generation)\n"
        "- Uses **FAISS** vector DB\n"
        "- Uses **Local LLM (Ollama)**\n"
        "- Uses **Local Embeddings (nomic-embed-text)**\n"
        "- Includes **citations** per answer\n"
        "- Hallucination control: **context-only prompt**\n"
    )

    st.subheader("✅ Tips")
    st.markdown(
        "- Always click **Build Knowledge Base** after uploading new PDFs\n"
        "- If the PDF is encrypted, it may be skipped\n"
        "- If answer is missing in PDFs, assistant should say: *I don't know based on the provided documents.*\n"
    )
