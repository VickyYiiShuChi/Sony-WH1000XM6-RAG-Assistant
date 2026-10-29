import os
import sys
from pathlib import Path

import streamlit as st

# Load Groq API key from Streamlit Cloud Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# -----------------------------
# Project configuration
# -----------------------------

# Automatically locate the project folder
PROJECT_DIR = Path(__file__).resolve().parent

# Locate the manual PDF inside the data folder
PDF_PATH = PROJECT_DIR / "data" / "Sony_WH1000XM6_UserManual.pdf"

os.chdir(PROJECT_DIR)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Sony WH-1000XM6 AI Assistant",
    page_icon="🎧",
    layout="wide",
)

st.title("🎧 Sony WH-1000XM6 AI Assistant")
st.caption(
    "Ask questions about your headphones using information "
    "retrieved from the official user manual."
)

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_status" not in st.session_state:
    st.session_state.index_status = "Checking existing index..."

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

# Check whether a usable ChromaDB index exists
@st.cache_resource
def check_index():
    from src.vector_store import VectorStore
    store = VectorStore()
    collection = store.create_collection()
    return collection.count()

try:
    document_count = check_index()
    if document_count > 0:
        st.session_state.index_ready = True
        st.session_state.index_status = (
            f"Index ready: {document_count} chunks loaded."
        )
    else:
        st.session_state.index_ready = False
        st.session_state.index_status = (
            "No indexed documents found. Build the index first."
        )
except Exception as exc:
    st.session_state.index_ready = False
    st.session_state.index_status = f"Index check failed: {exc}"

# -----------------------------
# Lazy-load the RAG pipeline
# -----------------------------
@st.cache_resource
def get_pipeline():
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Knowledge Base")

    st.write("**PDF manual:**")
    st.code(PDF_PATH.name)

    if PDF_PATH.exists():
        st.success("User manual PDF found.")
    else:
        st.error(f"PDF not found: {PDF_PATH}")

    if st.button(
        "Build / Rebuild Index",
        type="primary",
        use_container_width=True,
    ):
        if not PDF_PATH.exists():
            st.error(
                "Please check that the PDF exists at:\n"
                f"{PDF_PATH}"
            )
        else:
            try:
                with st.spinner(
                    "Loading pipeline and rebuilding the index..."
                ):
                    pipeline = get_pipeline()
                    result = pipeline.build_index(str(PDF_PATH))

                st.session_state.index_ready = True
                st.session_state.index_status = (
                    "Indexing completed successfully."
                )

                st.success("Indexing completed!")
                st.write(f"**Pages extracted:** {result['pages']}")
                st.write(f"**Sections detected:** {result['sections']}")
                st.write(f"**Chunks created:** {result['chunks']}")
                st.write(
                    f"**Chunks stored:** {result['indexed_chunks']}"
                )

            except Exception as exc:
                st.session_state.index_status = (
                    f"Indexing failed: {exc}"
                )
                st.error(st.session_state.index_status)

    st.divider()
    st.subheader("Index Status")
    st.info(st.session_state.index_status)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# Display chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['chapter']}** / "
                        f"{source['section']} "
                        f"(PDF pages {source['page_start']}-"
                        f"{source['page_end']})"
                    )

# -----------------------------
# Chat input
# -----------------------------
question = st.chat_input(
    "Ask a question about your Sony WH-1000XM6..."
)

if question:
    if not st.session_state.index_ready:
        st.warning(
            "The knowledge base is not ready yet. "
            "Please check the index status in the sidebar."
        )
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Loading models if needed and generating answer..."
            ):
                pipeline = get_pipeline()
                result = pipeline.ask(question)

            answer = result["answer"]
            sources = []

            for item in result.get("sources", []):
                meta = item.get("metadata", {})

                sources.append({
                    "chapter": meta.get("chapter", "Unknown chapter"),
                    "section": meta.get("section", "Unknown section"),
                    "page_start": meta.get("page_start", "?"),
                    "page_end": meta.get("page_end", "?"),
                })

            st.markdown(answer)

            if sources:
                with st.expander("Retrieved sources"):
                    for source in sources:
                        st.markdown(
                            f"- **{source['chapter']}** / "
                            f"{source['section']} "
                            f"(PDF pages {source['page_start']}-"
                            f"{source['page_end']})"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as exc:
            error_message = f"Unable to answer the question: {exc}"
            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": [],
            })
