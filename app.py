
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

RUNTIME_DIR = Path(tempfile.gettempdir()) / "sony_rag_assistant"
CHROMA_DIR = RUNTIME_DIR / "chroma_db"

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Set before importing project modules.
os.environ["SONY_RAG_RUNTIME_DIR"] = str(RUNTIME_DIR)

DATA_DIR = PROJECT_DIR / "data"
DEFAULT_PDF = DATA_DIR / "Sony_WH1000XM6_UserManual.pdf"

# Load Groq API key from Streamlit Secrets.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    # Local development can use an environment variable.
    pass


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Sony WH-1000XM6 Assistant",
    page_icon="🎧",
    layout="wide",
)


# -----------------------------
# Custom styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1450px;
    }

    .hero {
        text-align: center;
        padding: 1.2rem 0 2rem 0;
    }

    .hero h1 {
        font-size: 2.6rem;
        font-weight: 750;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.1rem;
        color: #888888;
        margin-top: 0;
    }

    .source-note {
        color: #888888;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 2.7rem;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎧 Meet Your Sony WH-1000XM6</h1>
        <p>Your headphones, your questions. Let's figure it out.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# -----------------------------
# Initialize RAG pipeline
# -----------------------------
@st.cache_resource
def get_ready_pipeline():
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()

    collection = pipeline.vector_store.create_collection()
    count = collection.count()

    if count == 0:
        if not DEFAULT_PDF.is_file():
            raise FileNotFoundError(
                f"User manual not found: {DEFAULT_PDF}"
            )

        # Build the index if it does not exist or is empty.
        pipeline.build_index(str(DEFAULT_PDF))

        collection = pipeline.vector_store.create_collection()
        count = collection.count()

        if count == 0:
            raise RuntimeError(
                "Indexing completed, but no searchable chunks were stored."
            )

    return pipeline, count


# -----------------------------
# Load knowledge base
# -----------------------------
try:
    with st.spinner("Getting everything ready..."):
        pipeline, chunk_count = get_ready_pipeline()

except Exception as exc:
    st.error("The assistant couldn't start properly.")
    st.caption(f"{type(exc).__name__}: {exc}")
    st.info("Please check the Streamlit deployment logs.")
    st.stop()


# -----------------------------
# Main layout
# -----------------------------
info_col, chat_col = st.columns(
    [1, 2.4],
    gap="large",
)

example_questions = [
    "How do I turn on noise cancelling?",
    "How can I connect to two devices?",
    "How do I check the battery level?",
    "What should I do if my headphones won't connect?",
]


# -----------------------------
# Left panel: Introduction
# -----------------------------
with info_col:
    st.markdown("### ✨ Explore your headphones")

    st.write(
        "Your guide to getting more out of your "
        "Sony WH-1000XM6 headphones."
    )

    st.markdown("#### 📖 Knowledge source")

    st.markdown(
        """
        <div class="source-note">
        Answers are based on the Sony WH-1000XM6 official
        user manual. Check the cited sections and PDF pages
        to verify the information.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### 💡 Try asking")

    for i, example in enumerate(example_questions):
        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example
            st.rerun()

    st.divider()

    st.caption("Sony WH-1000XM6 AI Assistant")

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# -----------------------------
# Right panel: Chat history
# -----------------------------
with chat_col:
    st.markdown("### 💬 Chat with your headphones guide")

    if not st.session_state.messages:
        st.info(
            "Welcome! Choose a suggested question on the left "
            "or type your own question in the box below."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📚 View sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"- **{source['chapter']}** / "
                            f"{source['section']} "
                            f"(PDF pages {source['page_start']}-"
                            f"{source['page_end']})"
                        )


# -----------------------------
# Bottom chat input
# -----------------------------
# st.bottom keeps the input at the bottom of the main app area.
# Keep it outside columns so it does not jump to the top of a column.
question = st.bottom.chat_input(
    "Ask about your headphones..."
)

# Handle a clicked example question or a typed question.
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with chat_col:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding an answer..."):
                    result = pipeline.ask(question)

                    answer = result["answer"]
                    sources = []

                    for item in result.get("sources", []):
                        meta = item.get("metadata", {})

                        sources.append(
                            {
                                "chapter": meta.get(
                                    "chapter", "Unknown chapter"
                                ),
                                "section": meta.get(
                                    "section", "Unknown section"
                                ),
                                "page_start": meta.get("page_start", "?"),
                                "page_end": meta.get("page_end", "?"),
                            }
                        )

                st.markdown(answer)

                if sources:
                    with st.expander("📚 View sources"):
                        for source in sources:
                            st.markdown(
                                f"- **{source['chapter']}** / "
                                f"{source['section']} "
                                f"(PDF pages {source['page_start']}-"
                                f"{source['page_end']})"
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                error_message = (
                    "Sorry, I couldn't answer that just now. "
                    f"({type(exc).__name__}: {exc})"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )

    # Rerun to display the latest messages in the chat history.
    st.rerun()
