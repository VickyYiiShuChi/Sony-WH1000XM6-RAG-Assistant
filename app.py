
import os
import sys
import tempfile
import time
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
    page_title="Sony Noise Cancelling Headphones WH-1000XM6",
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

    .block-container {
        max-width: 1600px;
        padding: 1.3rem 1.2rem 2rem 0.7rem;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 0.4rem 0 1.8rem;
    }

    .hero h1 {
        font-size: 2.15rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        line-height: 1.3;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: var(--muted);
        font-size: 0.96rem;
        margin: 0;
    }

    .hero-line {
        width: 46px;
        height: 4px;
        border-radius: 10px;
        background: var(--accent);
        margin: 0.9rem auto 0;
    }

    /* Section headings */
    .eyebrow {
        color: var(--accent);
        font-size: 0.73rem;
        font-weight: 750;
        letter-spacing: 1.1px;
        margin-bottom: 0.4rem;
    }

    .panel-title {
        font-size: 1.2rem;
        font-weight: 750;
        line-height: 1.4;
        margin-bottom: 0.35rem;
    }

    .panel-description {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.55;
        margin-bottom: 1.2rem;
    }

    /* Welcome card */
    .welcome-card {
        border: 1px solid var(--accent-border);
        border-radius: 14px;
        padding: 1.2rem;
        background: var(--accent-light);
        margin: 0.4rem 0 1.2rem;
    }

    .welcome-title {
        color: #40366F;
        font-size: 1rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .welcome-text {
        color: #625B7D;
        font-size: 0.87rem;
        line-height: 1.6;
    }

    /* Suggested question buttons */
    div[data-testid="stButton"] > button {
        min-height: 2.85rem;
        border-radius: 10px;
        text-align: left;
        white-space: normal;
        line-height: 1.4;
        font-size: 0.87rem;
        padding: 0.6rem 0.8rem;
        transition: border-color 0.2s ease,
                    color 0.2s ease,
                    background-color 0.2s ease;
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
        animation: messageFadeIn 0.35s ease-out;
    }

    @keyframes messageFadeIn {
        from {
            opacity: 0;
            transform: translateY(7px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
        border-color: var(--border);
    }

    div[data-testid="stChatInput"] {
        width: 100%;
        max-width: 100%;
    }

    div[data-testid="stChatInput"] textarea {
        border-radius: 12px;
    }

    hr {
        border-color: var(--border);
    }

    /* Mobile */
    @media (max-width: 800px) {
        .block-container {
            padding: 1rem 0.8rem 2rem;
        }

        .hero h1 {
            font-size: 1.65rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        div[data-testid="stChatMessage"] {
            animation: none;
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
        <h1>🎧 Sony Noise Cancelling Headphones WH-1000XM6</h1>
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
    with st.spinner("Preparing your assistant..."):
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
        '<div class="panel-title">Suggested questions</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-description">'
        'Choose a question to get started.'
        '</div>',
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
        "Information based on the official user manual "
        "for Sony Noise Cancelling Headphones WH-1000XM6."
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
# 11. Right panel: Chat history
# ==========================================
with right_col:
    st.markdown(
        '<div class="eyebrow">YOUR ASSISTANT</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-title">Chat</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-title">Hi there!</div>
                <div class="welcome-text">
                    Ask me anything about Sony Noise Cancelling
                    Headphones WH-1000XM6.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display previous messages
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

    # Keep the textbox within the chat column.
    typed_question = st.chat_input(
        "Ask about Sony Noise Cancelling Headphones WH-1000XM6..."
    )


# ==========================================
# 12. Smooth answer streaming
# ==========================================
def stream_answer(text):
    """Reveal the completed answer gradually."""
    words = text.split(" ")

    for index, word in enumerate(words):
        chunk = word if index == 0 else " " + word
        yield chunk

        # Short pause creates a gentle typing effect.
        # Longer answers are streamed a little faster.
        if len(words) < 80:
            time.sleep(0.018)
        else:
            time.sleep(0.008)


# ==========================================
# 13. Process typed or suggested question
# ==========================================
question = typed_question

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


if question:
    # Save the user question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Show the user's message and a gentle loading indicator.
    with right_col:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.status(
                    "Searching the user manual...",
                    expanded=False,
                ) as status:
                    result = pipeline.ask(question)
                    status.update(
                        label="Answer ready",
                        state="complete",
                        expanded=False,
                    )

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

                # Stream the answer into the chat.
                st.write_stream(stream_answer(answer))

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

    # Refresh so the conversation is displayed consistently.
    st.rerun()
