
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st


# ==========================================
# 1. Project configuration
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
# 2. Page configuration
# ==========================================
st.set_page_config(
    page_title="Sony WH-1000XM6 Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================
# 3. Custom styling
# ==========================================
st.markdown(
    """
    <style>
    :root {
        --accent: #7964D8;
        --accent-light: #F2EFFF;
        --accent-border: #E5DEFF;
        --muted: #77758A;
        --border: #E8E6F0;
    }

    /* Reduce left and right page margins */
    .block-container {
        max-width: 1600px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
        padding-left: 0.8rem;
        padding-right: 1.5rem;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 0.4rem 0 1.8rem 0;
    }

    .hero h1 {
        font-size: 2.25rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: var(--muted);
        font-size: 0.98rem;
        margin: 0;
    }

    .hero-line {
        width: 48px;
        height: 4px;
        border-radius: 10px;
        background: var(--accent);
        margin: 0.9rem auto 0;
    }

    /* Left panel */
    .eyebrow {
        color: var(--accent);
        font-size: 0.74rem;
        font-weight: 750;
        letter-spacing: 1.1px;
        margin-bottom: 0.45rem;
    }

    .panel-title {
        font-size: 1.2rem;
        font-weight: 750;
        line-height: 1.35;
        margin-bottom: 0.4rem;
    }

    .panel-description {
        color: var(--muted);
        font-size: 0.87rem;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    .intro-card {
        background: var(--accent-light);
        border: 1px solid var(--accent-border);
        border-radius: 14px;
        padding: 1rem;
        margin: 0.8rem 0 1.5rem;
    }

    .intro-icon {
        font-size: 1.5rem;
        margin-bottom: 0.4rem;
    }

    .intro-title {
        color: #40366F;
        font-size: 0.94rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .intro-text {
        color: #625B7D;
        font-size: 0.83rem;
        line-height: 1.65;
    }

    /* Buttons */
    div[data-testid="stButton"] > button {
        min-height: 2.85rem;
        border-radius: 10px;
        text-align: left;
        white-space: normal;
        line-height: 1.4;
        font-size: 0.87rem;
        padding: 0.6rem 0.8rem;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.65rem;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
        border-color: var(--border);
    }

    hr {
        border-color: var(--border);
    }

    /* Chat input is placed in the right column.
       Do not use page-level fixed positioning here. */
    div[data-testid="stChatInput"] {
        width: 100%;
        max-width: 100%;
    }

    div[data-testid="stChatInput"] textarea {
        border-radius: 12px;
    }

    /* Mobile */
    @media (max-width: 800px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero h1 {
            font-size: 1.7rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 4. Header
# ==========================================
st.markdown(
    """
    <div class="hero">
        <h1>🎧 Meet Your Sony WH-1000XM6</h1>
        <p>Your headphones, your questions. Let's figure it out.</p>
        <div class="hero-line"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 5. Session state
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ==========================================
# 6. Initialize RAG pipeline
# ==========================================
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


# ==========================================
# 7. Load knowledge base
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
# 8. Suggested questions
# ==========================================
example_questions = [
    "How do I turn on noise cancelling?",
    "How can I connect to two devices?",
    "How do I check the battery level?",
    "What should I do if my headphones won't connect?",
]


# ==========================================
# 9. Main layout
# ==========================================
# Left panel is compact; right panel has more room.
left_col, right_col = st.columns(
    [0.95, 2.5],
    gap="large",
)


# ==========================================
# 10. Left panel
# ==========================================
with left_col:
    st.markdown(
        '<div class="eyebrow">GET STARTED</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-title">Explore your headphones</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel-description">
            Find answers to common questions or choose a topic below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-card">
            <div class="intro-icon">🎧</div>
            <div class="intro-title">Your headphone guide</div>
            <div class="intro-text">
                Get help with noise cancelling, Bluetooth,
                battery life, touch controls and troubleshooting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Popular questions**")

    for i, example in enumerate(example_questions):
        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example
            st.rerun()

    st.markdown("<div style='height: 0.5rem'></div>",
                unsafe_allow_html=True)

    st.divider()

    st.caption("📘 Based on the official Sony user manual.")

    if st.button(
        "Clear conversation",
        key="clear_conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# ==========================================
# 11. Right panel: Chat history and input
# ==========================================
with right_col:
    st.markdown(
        '<div class="eyebrow">YOUR ASSISTANT</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-title">💬 Chat</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel-description">
            Ask a question about your Sony WH-1000XM6.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("### 👋 Hi there!")

            st.markdown(
                "I'm here to help you get the most out of your "
                "Sony WH-1000XM6 headphones."
            )

            st.markdown(
                "Choose a suggested question on the left, "
                "or type your own question below."
            )

    # Conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            sources = message.get("sources", [])

            if message["role"] == "assistant" and sources:
                with st.expander("📚 View sources"):
                    for source in sources:
                        st.markdown(
                            f"- **{source['chapter']}** / "
                            f"{source['section']} "
                            f"(PDF pages {source['page_start']}-"
                            f"{source['page_end']})"
                        )

    # Input stays within the right column.
    # It will appear below the chat history rather than being
    # artificially shifted across the entire page.
    typed_question = st.chat_input(
        "Ask about your headphones..."
    )


# ==========================================
# 12. Process typed or suggested question
# ==========================================
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
