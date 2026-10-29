
# ==========================================
# Enhanced styling
# ==========================================
st.markdown(
    """
    <style>
    .stApp {
        background: #f7f7fb;
    }

    .block-container {
        max-width: 1500px;
        padding: 1.5rem 2rem 5rem 1.5rem;
    }

    /* Header */
    .hero {
        text-align: left;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(120deg, #ffffff, #f0edff);
        border: 1px solid #e9e5fa;
        border-radius: 18px;
    }

    .hero-tag {
        display: inline-block;
        color: #6d4aff;
        background: #eee9ff;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 650;
        margin-bottom: 0.7rem;
    }

    .hero h1 {
        font-size: 2.15rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        margin: 0 0 0.4rem 0;
        color: #242137;
    }

    .hero p {
        color: #77738a;
        font-size: 0.98rem;
        margin: 0;
    }

    /* Main columns */
    div[data-testid="stHorizontalBlock"] {
        align-items: flex-start;
        gap: 1.5rem;
    }

    /* Left panel */
    .side-panel {
        padding: 1.2rem 1rem;
        background: #ffffff;
        border: 1px solid #e9e8f0;
        border-radius: 16px;
        margin-bottom: 1rem;
    }

    .side-title {
        font-size: 1.08rem;
        font-weight: 750;
        color: #29263a;
        margin-bottom: 0.35rem;
    }

    .side-description {
        font-size: 0.86rem;
        color: #858194;
        line-height: 1.6;
    }

    .mini-label {
        color: #7d6bc1;
        font-size: 0.74rem;
        font-weight: 750;
        letter-spacing: 0.8px;
        margin: 1.2rem 0 0.6rem 0;
    }

    /* Buttons */
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        min-height: 2.9rem;
        text-align: left;
        font-size: 0.87rem;
        border: 1px solid #e8e6f0;
        background: #ffffff;
        color: #39354a;
        transition: all 0.15s ease;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: #a99aff;
        color: #6344df;
        background: #f6f3ff;
    }

    /* Chat panel */
    .chat-heading {
        font-size: 1.15rem;
        font-weight: 750;
        color: #29263a;
        margin-bottom: 0.2rem;
    }

    .chat-subheading {
        color: #898596;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #eceaf3;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
        background: #ffffff;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #e8e5f2;
        border-radius: 10px;
        background: #ffffff;
    }

    /* Bottom chat input */
    div[data-testid="stBottom"] {
        width: 70% !important;
        left: 30% !important;
        right: 0 !important;
        padding: 0.6rem 2rem 0.8rem 1rem !important;
        box-sizing: border-box !important;
        background: rgba(247, 247, 251, 0.96);
    }

    div[data-testid="stBottom"] div[data-testid="stChatInput"] {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
    }

    /* Responsive */
    @media (max-width: 800px) {
        .block-container {
            padding: 1rem 0.8rem 5rem 0.8rem;
        }

        .hero {
            padding: 1rem;
        }

        .hero h1 {
            font-size: 1.65rem;
        }

        div[data-testid="stBottom"] {
            width: 100% !important;
            left: 0 !important;
            padding: 0.5rem 0.8rem !important;
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
        <div class="hero-tag">YOUR PERSONAL HEADPHONE ASSISTANT</div>
        <h1>🎧 Meet Your Sony WH-1000XM6</h1>
        <p>Your headphones, your questions. Let's figure it out.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# Main layout
# ==========================================
left_col, right_col = st.columns(
    [1, 2.25],
    gap="large",
)

example_questions = [
    "How do I turn on noise cancelling?",
    "How can I connect to two devices?",
    "How do I check the battery level?",
    "What should I do if my headphones won't connect?",
]

# Left panel
with left_col:
    st.markdown(
        """
        <div class="side-panel">
            <div class="side-title">✨ What can I help with?</div>
            <div class="side-description">
                Explore your headphones' features, settings,
                connectivity and troubleshooting.
            </div>
            <div class="mini-label">QUICK START</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, question_text in enumerate(example_questions):
        if st.button(
            question_text,
            key=f"example_{i}",
            use_container_width=True,
        ):
            st.session_state.pending_question = question_text
            st.rerun()

    st.markdown(
        """
        <div class="side-panel" style="margin-top: 1rem;">
            <div class="side-title">📖 Official manual</div>
            <div class="side-description">
                Answers are based on the Sony WH-1000XM6
                user manual. Open the source references to
                check the relevant PDF pages.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🗑️ Clear conversation",
        key="clear_conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# Right panel
with right_col:
    st.markdown(
        '<div class="chat-heading">💬 Conversation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="chat-subheading">'
        'Ask a question and get guidance from the official manual.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.info(
            "👋 Welcome! Select a question on the left "
            "or type your own question below to get started."
        )

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
