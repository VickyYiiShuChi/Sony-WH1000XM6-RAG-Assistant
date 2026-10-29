
import os
import tempfile
from pathlib import Path

import streamlit as st

# --------------------------------------------------
# 1. Project and runtime configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = PROJECT_DIR / "data" / "Sony_WH1000XM6_UserManual.pdf"

RUNTIME_DIR = Path(
    os.getenv(
        "SONY_RAG_RUNTIME_DIR",
        str(Path(tempfile.gettempdir()) / "sony_rag_assistant"),
    )
)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

os.environ["SONY_RAG_RUNTIME_DIR"] = str(RUNTIME_DIR)

# Load the API key from Streamlit Cloud Secrets or local environment
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from src.rag_pipeline import RAGPipeline


# --------------------------------------------------
# 2. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Sony WH-1000XM6 Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# 3. CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Use more of the available screen width */
    .block-container {
        max-width: 100% !important;
        padding-top: 1.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        padding-bottom: 6rem !important;
    }

    /* Centered page heading */
    .main-title {
        text-align: center;
        font-size: 2.35rem;
        font-weight: 750;
        margin-top: 0.2rem;
        margin-bottom: 0.35rem;
    }

    .main-subtitle {
        text-align: center;
        color: #8b8b8b;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Left introduction panel */
    .intro-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .intro-description {
        color: var(--text-color);
        line-height: 1.65;
        margin-bottom: 1rem;
    }

    .manual-note {
        background: var(--secondary-background-color);
        border-radius: 12px;
        padding: 1rem;
        line-height: 1.6;
        font-size: 0.92rem;
        margin-top: 1rem;
        margin-bottom: 1.3rem;
    }

    /* Make the columns start near the left edge */
    div[data-testid="stHorizontalBlock"] {
        align-items: flex-start;
        gap: 2rem;
    }

    /* Example question buttons */
    div[data-testid="stColumn"]:first-child
    div[data-testid="stButton"] > button {
        width: 100%;
        min-height: 2.8rem;
        text-align: left;
        justify-content: flex-start;
        border-radius: 10px;
        white-space: normal;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        padding: 0.5rem 0;
    }

    /* Keep the bottom chat input on the right side.
       Streamlit pins this element to the viewport bottom. */
    div[data-testid="stBottom"] {
        width: 66% !important;
        left: 34% !important;
        right: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1.25rem !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stBottom"] div[data-testid="stChatInput"] {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        box-sizing: border-box !important;
    }

    /* Small screens: use the full width for the input */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .main-title {
            font-size: 1.8rem;
        }

        div[data-testid="stBottom"] {
            width: 100% !important;
            left: 0 !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 4. Page heading
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎧 Meet Your Sony WH-1000XM6</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subtitle">'
    "Your headphones, your questions. Let's figure it out."
    "</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 5. Initialize the RAG pipeline
# --------------------------------------------------

@st.cache_resource
def get_ready_pipeline():
    pipeline = RAGPipeline()

    collection = pipeline.vector_store.collection
    count = collection.count()

    if count == 0:
        if not DEFAULT_PDF.exists():
            raise FileNotFoundError(
                f"Manual PDF not found: {DEFAULT_PDF}"
            )

        pipeline.build_index(str(DEFAULT_PDF))

        collection = pipeline.vector_store.collection
        count = collection.count()

    if count == 0:
        raise RuntimeError(
            "The knowledge base is empty after indexing the PDF."
        )

    return pipeline, count


try:
    pipeline, document_count = get_ready_pipeline()
except Exception as exc:
    st.error(f"Unable to initialize the assistant: {exc}")
    st.info(
        "Check that the PDF exists in the data folder, "
        "the Groq API key is configured, and dependencies are installed."
    )
    st.stop()


# --------------------------------------------------
# 6. Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------
# 7. Main layout
# --------------------------------------------------

info_col, chat_col = st.columns(
    [1, 2.4],
    gap="large",
)

# ---------------- Left panel ----------------

with info_col:
    st.markdown(
        '<div class="intro-title">✨ Explore your headphones</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="intro-description">
            Your guide to getting more out of your
            Sony WH-1000XM6 headphones.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="manual-note">
            <strong>📘 Official user manual</strong><br>
            Answers are based on the Sony WH-1000XM6
            official user manual. Check the cited sections
            and PDF pages to verify the information.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**💡 Try asking**")

    example_questions = [
        "How do I turn on noise cancelling?",
        "How can I connect to two devices?",
        "How do I check the battery level?",
        "What should I do if my headphones won't connect?",
    ]

    for index, example in enumerate(example_questions):
        if st.button(
            example,
            key=f"example_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example
            st.rerun()

    st.write("")

    if st.button(
        "🗑️ Clear conversation",
        key="clear_conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# ---------------- Right chat panel ----------------

with chat_col:
    st.subheader("💬 Chat with your headphones assistant")

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="manual-note">
                Ask a question about your headphones to get started.
                I'll help you find the relevant instructions
                and show the available source references.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
                    with st.expander("📚 View sources"):
                        for source_index, source in enumerate(sources, 1):
                            chapter = source.get("chapter", "")
                            section = source.get("section", "")
                            page_start = source.get("page_start")
                            page_end = source.get("page_end")

                            details = []

                            if chapter:
                                details.append(str(chapter))

                            if section:
                                details.append(str(section))

                            if page_start is not None:
                                if page_end is not None and page_end != page_start:
                                    details.append(
                                        f"PDF pages {page_start}–{page_end}"
                                    )
                                else:
                                    details.append(
                                        f"PDF page {page_start}"
                                    )

                            source_title = " · ".join(details) or "Manual reference"

                            st.markdown(
                                f"**Source {source_index}:** {source_title}"
                            )


# --------------------------------------------------
# 8. Process questions
# --------------------------------------------------

# Use Streamlit's bottom container so the input stays at the bottom.
with st.bottom:
    question = st.chat_input("Ask about your headphones...")

if question:
    st.session_state.pending_question = question

# Process either a typed question or a clicked example
question_to_process = st.session_state.pending_question

if question_to_process:
    st.session_state.pending_question = None

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question_to_process,
        }
    )

    with chat_col:
        with st.chat_message("user"):
            st.markdown(question_to_process)

        with st.chat_message("assistant"):
            with st.spinner("Looking through the manual..."):
                try:
                    result = pipeline.ask(question_to_process)

                    # Support either a plain-text answer or a dictionary result.
                    if isinstance(result, dict):
                        answer = (
                            result.get("answer")
                            or result.get("response")
                            or result.get("result")
                            or "I couldn't generate an answer for this question."
                        )

                        raw_sources = (
                            result.get("sources")
                            or result.get("source_documents")
                            or []
                        )
                    else:
                        answer = str(result)
                        raw_sources = []

                    sources = []

                    for item in raw_sources:
                        if isinstance(item, dict):
                            metadata = item.get("metadata", item)
                            sources.append(
                                {
                                    "chapter": metadata.get("chapter", ""),
                                    "section": metadata.get("section", ""),
                                    "page_start": metadata.get("page_start"),
                                    "page_end": metadata.get("page_end"),
                                }
                            )
                        else:
                            metadata = getattr(item, "metadata", {}) or {}
                            sources.append(
                                {
                                    "chapter": metadata.get("chapter", ""),
                                    "section": metadata.get("section", ""),
                                    "page_start": metadata.get("page_start"),
                                    "page_end": metadata.get("page_end"),
                                }
                            )

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 View sources"):
                            for source_index, source in enumerate(sources, 1):
                                details = []

                                if source.get("chapter"):
                                    details.append(source["chapter"])

                                if source.get("section"):
                                    details.append(source["section"])

                                page_start = source.get("page_start")
                                page_end = source.get("page_end")

                                if page_start is not None:
                                    if page_end is not None and page_end != page_start:
                                        details.append(
                                            f"PDF pages {page_start}–{page_end}"
                                        )
                                    else:
                                        details.append(
                                            f"PDF page {page_start}"
                                        )

                                source_title = (
                                    " · ".join(details) or "Manual reference"
                                )

                                st.markdown(
                                    f"**Source {source_index}:** {source_title}"
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
                        "Sorry, something went wrong while answering "
                        "your question. Please try again."
                    )

                    st.error(error_message)
                    st.caption(str(exc))

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        }
                    )

    st.rerun()
