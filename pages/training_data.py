import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


def chunk_text_sliding_window(text, chunk_size=2000, overlap=200):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_training_data(data):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Generate Q&A only from the provided data, without adding any extra information."),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    all_responses = []
    chunks = chunk_text_sliding_window(data, chunk_size=2000, overlap=200)
    for chunk in chunks:
        response = chain.invoke({"input": chunk})
        all_responses.append(response.content)
    final_output = "\n".join(all_responses)
    st.session_state.training_data = final_output
    return final_output

st.title("Generated Training Data")
if "training_data" in st.session_state:
    st.write(st.session_state.training_data)
else:
    st.warning("No training data found. Please Upload the data first.")
