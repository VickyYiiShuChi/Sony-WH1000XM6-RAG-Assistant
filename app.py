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
# Left panel: About + examples
# -----------------------------
with info_col:
    st.markdown("### ✨ Explore your headphones")

    st.markdown(
        """
        Ask about features, connectivity, controls, charging,
        and troubleshooting.

        **Source**

        Answers are based on the official Sony WH-1000XM6
        user manual, with source references available for checking.
        """
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

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# -----------------------------
# Right panel: Chat
# -----------------------------
with chat_col:
    st.markdown("### 💬 Chat with your headphones guide")

    if not st.session_state.messages:
        st.info(
            "Welcome! Choose a suggested question on the left "
            "or type your own question below."
        )

    # Display conversation history
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

    # Keep the chat input at the bottom of the page
    typed_question = st.chat_input(
        "Ask about your headphones..."
    )

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
                                "page_start": meta.get(
                                    "page_start", "?"
                                ),
                                "page_end": meta.get(
                                    "page_end", "?"
                                ),
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
