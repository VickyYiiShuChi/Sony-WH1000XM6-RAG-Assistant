
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
        --accent-light: #F1EEFF;
        --muted: #77758A;
        --border: #E8E6F0;
    }

    /* Give the page more room on the left and right */
    .block-container {
        max-width: 1550px;
        padding-top: 1.4rem;
        padding-bottom: 6rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 0.5rem 0 2rem 0;
    }

    .hero h1 {
        font-size: 2.35rem;
        font-weight: 750;
        letter-spacing: -0.9px;
        margin-bottom: 0.45rem;
        line-height: 1.25;
    }

    .hero p {
        font-size: 1rem;
        color: var(--muted);
        margin: 0;
    }

    .hero-line {
        width: 54px;
        height: 4px;
        border-radius: 10px;
        background: var(--accent);
        margin: 1rem auto 0 auto;
    }

    /* Column headings */
    .eyebrow {
        color: var(--accent);
        font-size: 0.75rem;
        font-weight: 750;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .panel-title {
        font-size: 1.25rem;
        font-weight: 750;
        letter-spacing: -0.3px;
        margin-bottom: 0.35rem;
    }

    .panel-description {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.6;
        margin-bottom: 1.3rem;
    }

    /* Intro card */
    .intro-card {
        background: var(--accent-light);
        border: 1px solid #E5DEFF;
        border-radius: 15px;
        padding: 1.1rem 1rem;
        margin: 0.8rem 0 1.5rem 0;
    }

    .intro-icon {
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
    }

    .intro-title {
        font-size: 0.95rem;
        font-weight: 750;
        color: #40366F;
        margin-bottom: 0.35rem;
    }

    .intro-text {
        color: #625B7D;
        font-size: 0.82rem;
        line-height: 1.65;
    }

    /* Suggested question buttons */
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        min-height: 2.9rem;
        text-align: left;
        white-space: normal;
        font-size: 0.87rem;
        line-height: 1.4;
        transition: border-color 0.15s ease;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    /* Chat */
    div[data-testid="stChatMessage"] {
        border-radius: 13px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.7rem;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    /* Divider */
    hr {
        border-color: var(--border);
    }

    /* Bottom chat input: visually align with the right column */
    div[data-testid="stBottom"] {
        width: 71% !important;
        left: 29% !important;
        right: 0 !important;
        padding: 0.65rem 2.2rem 0.8rem 1rem !important;
        box-sizing: border-box !important;
        background: var(--background-color);
        border-top: 1px solid var(--border);
    }

    div[data-testid="stBottom"] div[data-testid="stChatInput"] {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        box-sizing: border-box !important;
    }

    /* Small screen adjustments */
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
# Wider left panel, spacious chat panel
left_col, right_col = st.columns(
    [1, 2.35],
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
            Choose a question below, or type your own
            question in the chat.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-card">
            <div class="intro-icon">🎧</div>
            <div class="intro-title">Your personal headphone guide</div>
            <div class="intro-text">
                Get help with settings, Bluetooth connections,
                battery life, touch controls, and troubleshooting.
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

    st.markdown("<div style='height: 0.6rem'></div>",
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
# 11. Right panel: Chat
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
        '<div class="panel-description">'
        'Ask a question and get an answer from the user manual.'
        '</div>',
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
                "Select a suggested question on the left, "
                "or ask me anything using the message box below."
            )

    # Display conversation history
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


# ==========================================
# 12. Bottom chat input
# ==========================================
with st.bottom:
    typed_question = st.chat_input(
        "Ask about your headphones..."
    )


# ==========================================
# 13. Process question
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
