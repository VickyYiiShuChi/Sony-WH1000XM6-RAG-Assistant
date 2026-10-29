
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

# Store the ChromaDB index in a writable runtime directory.
# Streamlit Cloud may clear this directory after a restart.
RUNTIME_DIR = Path(tempfile.gettempdir()) / "sony_rag_assistant"
CHROMA_DIR = RUNTIME_DIR / "chroma_db"

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Set this before importing project modules that read config.py.
os.environ["SONY_RAG_RUNTIME_DIR"] = str(RUNTIME_DIR)

# The manual is bundled with the project.
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_PDF = DATA_DIR / "Sony_WH1000XM6_UserManual.pdf"

# Load Groq API key from Streamlit Cloud Secrets.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    # Allows local development with an environment variable.
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


# -----------------------------
# Initialize the RAG pipeline
# -----------------------------
@st.cache_resource
def get_ready_pipeline():
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()

    # Check whether a searchable index already exists.
    collection = pipeline.vector_store.create_collection()
    count = collection.count()

    if count == 0:
        # Build the index automatically when it is missing or empty.
        if not DEFAULT_PDF.is_file():
            raise FileNotFoundError(
                "The default user manual was not found. "
                f"Expected location: {DEFAULT_PDF}"
            )

        pipeline.build_index(str(DEFAULT_PDF))

        # Verify that indexing produced searchable chunks.
        collection = pipeline.vector_store.create_collection()
        count = collection.count()

        if count == 0:
            raise RuntimeError(
                "Indexing completed, but no searchable chunks were stored."
            )

    return pipeline, count


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Knowledge Base")
    st.caption("Sony WH-1000XM6 User Manual")

    if DEFAULT_PDF.is_file():
        st.success("User manual available.")
        st.caption(DEFAULT_PDF.name)
    else:
        st.error("The bundled user manual PDF is missing.")

    st.divider()
    st.subheader("Index Status")


# -----------------------------
# Load the knowledge base
# -----------------------------
try:
    with st.spinner(
        "Preparing the knowledge base. "
        "The first startup may take a few minutes..."
    ):
        pipeline, chunk_count = get_ready_pipeline()

    index_status = f"Ready: {chunk_count} chunks indexed."

    with st.sidebar:
        st.success(index_status)

except Exception as exc:
    pipeline = None
    error_message = f"{type(exc).__name__}: {exc}"

    with st.sidebar:
        st.error("Knowledge base initialization failed.")
        st.caption(error_message)

    st.error(
        "The assistant could not prepare its knowledge base. "
        "Please check the deployment logs."
    )
    st.stop()


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
    "Ask a question about the Sony WH-1000XM6..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Retrieving relevant passages and generating an answer..."
            ):
                print(
                    "[RAG DEBUG] Question received:",
                    repr(question),
                    flush=True,
                )

                result = pipeline.ask(question)

                answer = result["answer"]
                sources = []

                for item in result.get("sources", []):
                    meta = item.get("metadata", {})

                    sources.append({
                        "chapter": meta.get(
                            "chapter", "Unknown chapter"
                        ),
                        "section": meta.get(
                            "section", "Unknown section"
                        ),
                        "page_start": meta.get("page_start", "?"),
                        "page_end": meta.get("page_end", "?"),
                    })

                print(
                    "[RAG DEBUG] Answer generated. "
                    f"Length: {len(answer)}",
                    flush=True,
                )

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


# -----------------------------
# Clear chat
# -----------------------------
with st.sidebar:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
