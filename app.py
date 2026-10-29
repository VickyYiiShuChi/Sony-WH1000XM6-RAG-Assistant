
import os
import tempfile
from pathlib import Path

import streamlit as st


# ==================================================
# 1. Configuration
# ==================================================

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

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from src.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="Sony WH-1000XM6 Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==================================================
# 2. UI Styling
# ==================================================

st.markdown(
    """
    <style>
    :root {
        --accent: #7664D8;
        --accent-light: #F0EDFF;
        --border: #E8E7EE;
        --muted: #777782;
    }

    /* Page width and spacing */
    .block-container {
        max-width: 1500px !important;
        padding: 1.5rem 2rem 7rem 2rem !important;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 0.5rem 0 2rem 0;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        line-height: 1.25;
        margin: 0;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin-top: 0.65rem;
    }

    /* Section headings */
    .panel-heading {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.2px;
        margin-bottom: 0.4rem;
    }

    .panel-description {
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.65;
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    /* Manual information */
    .manual-card {
        background: var(--secondary-background-color);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        margin: 1.1rem 0 1.5rem 0;
    }

    .manual-card-title {
        font-size: 0.93rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .manual-card-description {
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.6;
    }

    .examples-heading {
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0 0 0.65rem 0;
    }

    /* Example buttons */
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        min-height: 2.65rem;
        font-size: 0.88rem;
        transition: border-color 0.15s ease, background 0.15s ease;
    }

    /* Chat panel heading */
    .chat-heading {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
    }

    .chat-caption {
        color: var(--muted);
        font-size: 0.86rem;
        margin-bottom: 1.25rem;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.7rem;
    }

    /* Source details */
    div[data-testid="stExpander"] {
        border-radius: 10px;
        border-color: var(--border);
    }

    /* Bottom chat input: align with the right-hand column */
    div[data-testid="stBottom"] {
        width: 68% !important;
        left: 32% !important;
        right: 0 !important;
        padding: 0.65rem 2rem 0.85rem 1.25rem !important;
        box-sizing: border-box !important;
        background: var(--background-color);
    }

    div[data-testid="stBottom"] div[data-testid="stChatInput"] {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stBottom"] textarea {
        font-size: 0.95rem !important;
    }

    /* Clear conversation button */
    div[data-testid="stButton"] button[kind="secondary"] {
        border-color: var(--border);
    }

    /* Responsive layout */
    @media (max-width: 900px) {
        .block-container {
            padding: 1.1rem 1rem 6rem 1rem !important;
        }

        .hero-title {
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


# ==================================================
# 3. Session State
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ==================================================
# 4. Load RAG Pipeline
# ==================================================

@st.cache_resource
def get_ready_pipeline():
    pipeline = RAGPipeline()

    # Initialize the ChromaDB collection before counting
    collection = pipeline.vector_store.create_collection(reset=False)
    count = collection.count()

    # Build the index only when the collection is empty
    if count == 0:
        if not DEFAULT_PDF.exists():
            raise FileNotFoundError(
                f"Manual PDF not found: {DEFAULT_PDF}"
            )

        pipeline.build_index(str(DEFAULT_PDF))

        collection = pipeline.vector_store.create_collection(reset=False)
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
        "Check the PDF path, Groq API key, installed dependencies, "
        "and the RAG pipeline configuration."
    )
    st.stop()


# ==================================================
# 5. Header
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🎧 Meet Your Sony WH-1000XM6</div>
        <div class="hero-subtitle">
            Your headphones, your questions. Let's figure it out.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 6. Main Layout
# ==================================================

left_col, right_col = st.columns(
    [1, 2.25],
    gap="large",
    vertical_alignment="top",
)


# --------------------------------------------------
# Left: Introduction and examples
# --------------------------------------------------

with left_col:
    st.markdown(
        '<div class="eyebrow">YOUR HEADPHONE GUIDE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-heading">Explore your headphones</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel-description">
            Need help with a setting, connection, or feature?
            Ask a question and find the instructions you need.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="manual-card">
            <div class="manual-card-title">📘 Official user manual</div>
            <div class="manual-card-description">
                Answers are grounded in the Sony WH-1000XM6
                user manual. Open the source references to
                check the relevant sections and PDF pages.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="examples-heading">Try a question</div>',
        unsafe_allow_html=True,
    )

    example_questions = [
        ("🎧  Noise cancelling", "How do I turn on noise cancelling?"),
        ("📱  Connect devices", "How can I connect to two devices?"),
        ("🔋  Battery level", "How do I check the battery level?"),
        ("🔧  Troubleshooting", "What should I do if my headphones won't connect?"),
    ]

    for index, (label, question_text) in enumerate(example_questions):
        if st.button(
            label,
            key=f"example_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = question_text
            st.rerun()

    st.write("")

    if st.button(
        "Clear conversation",
        key="clear_conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# --------------------------------------------------
# Right: Chat
# --------------------------------------------------

with right_col:
    st.markdown(
        '<div class="chat-heading">Ask your assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="chat-caption">'
        'Ask in your own words. Source references are available with answers.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("### Welcome! 👋")
            st.markdown(
                "What would you like to know about your Sony WH-1000XM6?"
            )
            st.caption(
                "Choose an example on the left or type your own question below."
            )

    # Display the existing conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
                    with st.expander("📚 Sources and manual pages"):
                        for source_index, source in enumerate(sources, 1):
                            details = []

                            if source.get("chapter"):
                                details.append(str(source["chapter"]))

                            if source.get("section"):
                                details.append(str(source["section"]))

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

                            title = " · ".join(details) or "Manual reference"
                            st.markdown(f"**Source {source_index}:** {title}")


# ==================================================
# 7. Chat Input
# ==================================================

# Streamlit pins this input to the bottom of the page.
# CSS above visually aligns it with the right-hand chat column.
with st.bottom:
    typed_question = st.chat_input("Ask about your headphones...")


# ==================================================
# 8. Handle User Question
# ==================================================

if typed_question:
    st.session_state.pending_question = typed_question

question_to_process = st.session_state.pending_question

if question_to_process:
    st.session_state.pending_question = None

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question_to_process,
        }
    )

    try:
        with right_col:
            with st.chat_message("assistant"):
                with st.spinner("Searching the manual..."):
                    result = pipeline.ask(question_to_process)

                # Adapt to either a dictionary or plain-text pipeline result
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
                    with st.expander("📚 Sources and manual pages"):
                        for source_index, source in enumerate(sources, 1):
                            details = []

                            if source.get("chapter"):
                                details.append(str(source["chapter"]))

                            if source.get("section"):
                                details.append(str(source["section"]))

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

                            title = " · ".join(details) or "Manual reference"
                            st.markdown(f"**Source {source_index}:** {title}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

    except Exception as exc:
        error_message = (
            "Sorry, something went wrong while answering your question."
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "sources": [],
            }
        )

        st.error(error_message)
        st.caption(str(exc))

    st.rerun()
