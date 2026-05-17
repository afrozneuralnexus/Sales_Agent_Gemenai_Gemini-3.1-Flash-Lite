import os
import streamlit as st

from rag_pipeline import (
    create_vector_store,
    ask_question
)

st.set_page_config(
    page_title="Gemini RAG Chatbot",
    layout="wide"
)

st.title("📄 Gemini Flash-Lite RAG Chatbot")

st.write("Upload PDF and ask questions.")

uploaded_file = st.file_uploader(
    "Upload PDF File",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs("data", exist_ok=True)

    file_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Successfully")

    if st.button("Create Knowledge Base"):

        with st.spinner("Creating Vector Database..."):

            message = create_vector_store(file_path)

        st.success(message)

st.divider()

question = st.text_input(
    "Ask Question From PDF"
)

if st.button("Generate Answer"):

    if question.strip() == "":
        st.warning("Please enter question")
    else:

        with st.spinner("Thinking..."):

            answer = ask_question(question)

        st.subheader("Answer")
        st.write(answer)
