import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# -----------------------------
# Project configuration
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Load Groq API key from Streamlit Cloud Secrets.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    # Allows local development when Streamlit Secrets are not configured.
    pass

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
    "retrieved from the user manual."
)

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_status" not in st.session_state:
    st.session_state.index_status = "No indexed documents found."

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "uploaded_pdf_path" not in st.session_state:
    st.session_state.uploaded_pdf_path = None

# Use a writable directory for uploaded PDFs and ChromaDB.
# The temporary files remain available during the current app runtime.
RUNTIME_DIR = Path(tempfile.gettempdir()) / "sony_rag_assistant"
DATA_DIR = RUNTIME_DIR / "data"
CHROMA_DIR = RUNTIME_DIR / "chroma_db"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Set before importing modules that read CHROMA_DIR.
os.environ["SONY_RAG_RUNTIME_DIR"] = str(RUNTIME_DIR)

# -----------------------------
# Lazy-load the RAG pipeline
# -----------------------------
@st.cache_resource
def get_pipeline():
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline()

@st.cache_resource
def check_index():
    from src.vector_store import VectorStore
    store = VectorStore()
    collection = store.create_collection()
    return collection.count()

# -----------------------------
# Sidebar: Knowledge Base
# -----------------------------
with st.sidebar:
    st.header("Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload the user manual (PDF)",
        type=["pdf"],
        help="Upload a PDF manual to build a searchable knowledge base.",
    )

    if uploaded_file is not None:
        if uploaded_file.size > 20 * 1024 * 1024:
            st.error("The PDF is too large. Please upload a file under 20 MB.")
        else:
            upload_path = DATA_DIR / Path(uploaded_file.name).name
            upload_path.write_bytes(uploaded_file.getvalue())
            st.session_state.uploaded_pdf_path = str(upload_path)
            st.success(f"PDF uploaded: {uploaded_file.name}")

    pdf_path = st.session_state.uploaded_pdf_path

    if pdf_path and Path(pdf_path).is_file():
        st.success("User manual PDF found.")
        st.caption(Path(pdf_path).name)
    else:
        st.info("Upload a PDF to get started.")

    if st.button(
        "Build / Rebuild Index",
        type="primary",
        use_container_width=True,
    ):
        if not pdf_path or not Path(pdf_path).is_file():
            st.error("Please upload a PDF manual first.")
        else:
            try:
                with st.spinner(
                    "Building the knowledge base. This may take a few minutes..."
                ):
                    pipeline = get_pipeline()
                    result = pipeline.build_index(pdf_path)

                # Clear cached index count and verify the new index.
                check_index.clear()
                count = check_index()

                st.session_state.index_ready = count > 0
                st.session_state.index_status = (
                    f"Index ready: {count} chunks loaded."
                    if count > 0
                    else "Indexing completed, but no chunks were found."
                )

                st.success("Indexing completed!")
                st.write(f"**Pages extracted:** {result['pages']}")
                st.write(f"**Sections detected:** {result['sections']}")
                st.write(f"**Chunks created:** {result['chunks']}")
                st.write(f"**Chunks stored:** {result['indexed_chunks']}")

            except Exception as exc:
                st.session_state.index_ready = False
                st.session_state.index_status = (
                    f"Indexing failed: {type(exc).__name__}: {exc}"
                )
                st.error(st.session_state.index_status)

    st.divider()
    st.subheader("Index Status")

    # Refresh index count to reflect the persisted collection.
    try:
        count = check_index()
        if count > 0:
            st.session_state.index_ready = True
            st.session_state.index_status = f"Index ready: {count} chunks loaded."
        elif not st.session_state.index_ready:
            st.session_state.index_status = (
                "No indexed documents found. Upload a PDF and build the index."
            )
    except Exception as exc:
        st.session_state.index_ready = False
        st.session_state.index_status = f"Index check failed: {exc}"

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
                        f"- **{source['chapter']}** / {source['section']} "
                        f"(PDF pages {source['page_start']}-{source['page_end']})"
                    )

# -----------------------------
# Chat input
# -----------------------------
question = st.chat_input(
    "Ask a question about the uploaded manual..."
)

if question:
    if not st.session_state.index_ready:
        st.warning(
            "The knowledge base is not ready yet. "
            "Upload a PDF and build the index first."
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
            with st.spinner("Retrieving relevant passages and generating an answer..."):
                print(
                    "[RAG DEBUG] Question received:",
                    repr(question),
                    flush=True,
                )
                pipeline = get_pipeline()
                print(
                    "[RAG DEBUG] Pipeline initialized.",
                    flush=True,
                )
                result = pipeline.ask(question)
                print(
                    "[RAG DEBUG] Answer generated. "
                    f"Length: {len(result.get('answer', ''))}",
                    flush=True,
                )

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
                            f"- **{source['chapter']}** / {source['section']} "
                            f"(PDF pages {source['page_start']}-{source['page_end']})"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as exc:
            error_message = (
                f"Unable to answer the question: "
                f"{type(exc).__name__}: {exc}"
            )
            st.error(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": [],
            })
