import streamlit as st
import os
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import tempfile
from pages.training_data import get_training_data
load_dotenv()

VECTORSTORE_PATH = "vectorstore.index"

def get_text(pdf_docs):
    text = ""
    converter = DocumentConverter()
    for pdf in pdf_docs:
        file_extension = os.path.splitext(pdf.name)[1].lower()
        if file_extension == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf.read())
                tmp_file_path = tmp_file.name
            try:
                result = converter.convert(tmp_file_path)
                text += result.document.export_to_markdown()
            finally:
                os.remove(tmp_file_path)
        elif file_extension == ".docx":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(pdf.read())
                tmp_file_path = tmp_file.name
            try:
                result = converter.convert(tmp_file_path)
                text += result.document.export_to_markdown()
            finally:
                os.remove(tmp_file_path)
        else:
            st.warning(f"Unsupported file type: {file_extension}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks, existing_vectorstore=None):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large")
    if existing_vectorstore:
        existing_vectorstore.add_texts(text_chunks)
        existing_vectorstore.save_local(VECTORSTORE_PATH)
        return existing_vectorstore
    else:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        return vectorstore

def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large")
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def main():
    existing_vectorstore = load_vectorstore()
    if existing_vectorstore:
        st.session_state.vectorstore = existing_vectorstore
    st.title("Prof. Dr Deepak Ravindran")
    st.subheader("The UK's Go-To Doctor in Pain Management")
    files = st.file_uploader("Upload PDFs/DOCx here", accept_multiple_files=True)
    if st.button("Process Files") and files:
        with st.spinner("Processing Files..."):
            raw_text = get_text(files)
            training_data = get_training_data(raw_text)
            text_chunks = get_text_chunks(training_data)
            vectorstore = get_vectorstore(text_chunks, existing_vectorstore)
            st.session_state.vectorstore = vectorstore
            st.success("Uploaded Successfully!")

main()
