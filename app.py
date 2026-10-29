
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st


# ==========================================
# Project configuration
# ==========================================
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


# ==========================================
# Page configuration
# ==========================================
st.set_page_config(
    page_title="Sony WH-1000XM6 Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================
# Custom CSS
# ==========================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* Header */
    .hero {
        text-align: center;
        padding-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.6px;
        margin-bottom: 0.35rem;
    }

    .hero p {
        color: #888888;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Section headings */
    .section-heading {
        font-size: 1.05rem;
        font-weight: 650;
        margin-bottom: 0.3rem;
    }

    .helper-text {
        color: #888888;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    /* Suggested question buttons */
    div[data-testid="stButton"] > button {
        width: 100%;
        min-height: 2.7rem;
        border-radius: 8px;
        text-align: left;
        font-size: 0.87rem;
        white-space: normal;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.5rem;
    }

    /* Source expander */
    div[data-testid="stExpander"] {
        border-radius: 8px;
    }

    /* Align bottom input approximately with the right column */
    div[data-testid="stBottom"] {
        width: 70% !important;
        left: 30% !important;
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

    /* Mobile layout */
    @media (max-width: 800px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero h1 {
            font-size: 1.6rem;
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


# ==========================================
# Header
# ==========================================
st.markdown(
    """
    <div class="hero">
        <h1>🎧 Sony WH-1000XM6 Assistant</h1>
        <p>Your headphones, your questions. Let's figure it out.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# Session state
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ==========================================
# Initialize RAG pipeline
# ==========================================
@st.cache_resource
def get_ready_pipeline():
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()

    # Initialize the collection before checking its size.
    collection = pipeline.vector_store.create_collection(
        reset=False
    )
    count = collection.count()

    # Build the index only if the collection is empty.
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


# ==========================================
# Load knowledge base
# ==========================================
try:
    with st.spinner("Getting everything ready..."):
        pipeline, chunk_count = get_ready_pipeline()

except Exception as exc:
    st.error("The assistant couldn't start properly.")
    st.caption(f"{type(exc).__name__}: {exc}")
    st.info("Please check the deployment logs.")
    st.stop()


# ==========================================
# Suggested questions
# ==========================================
example_questions = [
    "How do I turn on noise cancelling?",
    "How can I connect to two devices?",
    "How do I check the battery level?",
    "What should I do if my headphones won't connect?",
]


# ==========================================
# Main layout
# Left: Suggested questions
# Right: Chat
# ==========================================
left_col, right_col = st.columns(
    [1, 2.3],
    gap="large",
)


# ==========================================
# Left column
# ==========================================
with left_col:
    st.markdown(
        '<div class="section-heading">Suggested questions</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="helper-text">Choose a question to get started.</div>',
        unsafe_allow_html=True,
    )

    for i, example in enumerate(example_questions):
        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example
            st.rerun()

    st.divider()

    st.caption(
        "Answers are based on the official Sony user manual."
    )

    if st.button(
        "Clear conversation",
        key="clear_conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# ==========================================
# Right column: Chat history
# ==========================================
with right_col:
    st.markdown("#### Chat")

    if not st.session_state.messages:
        st.caption(
            "Your conversation will appear here."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            sources = message.get("sources", [])

            if message["role"] == "assistant" and sources:
                with st.expander("View sources"):
                    for source in sources:
                        st.markdown(
                            f"- **{source['chapter']}** / "
                            f"{source['section']} "
                            f"(PDF pages {source['page_start']}-"
                            f"{source['page_end']})"
                        )


# ==========================================
# Bottom chat input
# ==========================================
# Requires a Streamlit version that supports st.bottom.
with st.bottom:
    typed_question = st.chat_input(
        "Ask about your headphones..."
    )


# ==========================================
# Process typed or suggested question
# ==========================================
question = typed_question

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


if question:
    # Save user question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:
        with st.spinner("Finding an answer..."):
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
                    "page_start": meta.get(
                        "page_start", "?"
                    ),
                    "page_end": meta.get(
                        "page_end", "?"
                    ),
                }
            )

        # Save assistant answer and source references.
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

    # Rerun to display the latest messages.
    st.rerun()
