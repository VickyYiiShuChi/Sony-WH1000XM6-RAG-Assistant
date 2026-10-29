
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

os.environ["SONY_RAG_RUNTIME_DIR"] = str(RUNTIME_DIR)

DEFAULT_PDF = (
    PROJECT_DIR / "data" / "Sony_WH1000XM6_UserManual.pdf"
)

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Sony WH-1000XM6 Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------
# Custom styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 6rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    .hero {
        text-align: center;
        padding: 0.5rem 0 2rem 0;
    }

    .hero h1 {
        font-size: 2.5rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        margin-bottom: 0.45rem;
    }

    .hero p {
        font-size: 1rem;
        color: #888888;
        margin-top: 0;
    }

    .section-heading {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .helper-text {
        color: #888888;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .manual-card {
        border: 1px solid #e8e7ee;
        border-radius: 12px;
        padding: 1rem;
        margin: 1.2rem 0 1.5rem 0;
        background: var(--secondary-background-color);
    }

    .manual-title {
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .manual-description {
        font-size: 0.83rem;
        color: #888888;
        line-height: 1.6;
    }

    div[data-testid="stHorizontalBlock"] {
        align-items: flex-start;
        gap: 2rem;
    }

    div[data-testid="stButton"] > button {
        border-radius: 10px;
        min-height: 2.8rem;
        font-size: 0.88rem;
        text-align: left;
        white-space: normal;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.6rem;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
    }

    /* Keep the bottom input aligned with the right chat column */
    div[data-testid="stBottom"] {
        width: 68% !important;
        left: 32% !important;
        right: 0 !important;
        padding: 0.5rem 1.5rem 0.75rem 1rem !important;
        box-sizing: border-box !important;
        background: var(--background-color);
    }

    div[data-testid="stBottom"] div[data-testid="stChatInput"] {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        box-sizing: border-box !important;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero h1 {
            font-size: 1.8rem;
        }

        div[data-testid="stBottom"] {
            width: 100% !important;
            left: 0 !important;
            padding: 0.5rem 1rem 0.75rem 1rem !important;
        }
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

    collection = pipeline.vector_store.create_collection(
        reset=False
    )
    count = collection.count()

    if count == 0:
        if not DEFAULT_PDF.is_file():
            raise FileNotFoundError(
                f"User manual not found: {DEFAULT_PDF}"
            )

        pipeline.build_index(str(DEFAULT_PDF))

        collection = pipeline.vector_store.create_collection(
            reset=False
        )
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
    st.info("Please check the deployment logs.")
    st.stop()


# -----------------------------
# Main layout
# Left: examples
# Right: chat
# -----------------------------
examples_col, chat_col = st.columns(
    [1, 2.4],
    gap="large",
)


# -----------------------------
# Left panel
# -----------------------------
with examples_col:
    st.markdown(
        '<div class="section-heading">✨ Explore your headphones</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="helper-text">
            Need help with a feature or setting?
            Start with one of these common questions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="manual-card">
            <div class="manual-title">📘 Official user manual</div>
            <div class="manual-description">
                Answers are based on the Sony WH-1000XM6
                official user manual. Check the source references
                and PDF pages to verify the information.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Suggested questions**")

    example_questions = [
        "How do I turn on noise cancelling?",
        "How can I connect to two devices?",
        "How do I check the battery level?",
        "What should I do if my headphones won't connect?",
    ]

    for i, example in enumerate(example_questions):
        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example
            st.rerun()

    st.divider()

    st.caption("Sony WH-1000XM6")

    st.caption(
        "Ask about features, connectivity, controls, "
        "charging, and troubleshooting."
    )

    if st.button(
        "🗑️ Clear conversation",
        key="clear_conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# -----------------------------
# Right panel: chat history
# -----------------------------
with chat_col:
    st.markdown("#### 💬 Chat")

    if not st.session_state.messages:
        st.markdown(
            """
            Welcome! Ask me anything about your headphones.
            You can also choose a suggested question on the left.
            """
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            sources = message.get("sources", [])

            if message["role"] == "assistant" and sources:
                with st.expander("📚 View sources"):
                    for source in sources:
                        chapter = source.get(
                            "chapter", "Unknown chapter"
                        )
                        section = source.get(
                            "section", "Unknown section"
                        )
                        page_start = source.get("page_start", "?")
                        page_end = source.get("page_end", "?")

                        st.markdown(
                            f"- **{chapter}** / {section} "
                            f"(PDF pages {page_start}-{page_end})"
                        )


# -----------------------------
# Chat input
# -----------------------------
# Page-level bottom input, visually aligned with the right column.
with st.bottom:
    typed_question = st.chat_input(
        "Ask about your headphones..."
    )


# -----------------------------
# Process typed or suggested question
# -----------------------------
question = typed_question

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

    try:
        result = pipeline.ask(question)

        answer = result["answer"]
        sources = []

        for item in result.get("sources", []):
            if isinstance(item, dict):
                meta = item.get("metadata", item)
            else:
                meta = getattr(item, "metadata", {}) or {}

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

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "sources": [],
            }
        )

    st.rerun()
