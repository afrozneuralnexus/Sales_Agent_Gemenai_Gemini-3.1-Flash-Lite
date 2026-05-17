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

st.title("📄 Gemini RAG Chatbot")

st.write("Upload PDF and ask questions from your document.")

uploaded_file = st.file_uploader(
    "Upload PDF",
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

    if st.button("Create Vector Store"):

        with st.spinner("Processing PDF..."):

            msg = create_vector_store(file_path)

        st.success(msg)

st.divider()

question = st.text_input(
    "Ask Question"
)

if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question")
    else:

        with st.spinner("Generating Answer..."):

            response = ask_question(question)

        st.subheader("Answer")
        st.write(response)
