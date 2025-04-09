import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


def get_training_data(data):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Generate Q&A only from the provided data, without adding any extra information."),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"input": data})
    st.session_state.training_data = response.content
    return response.content

st.title("Generated Training Data")
if "training_data" in st.session_state:
    st.write(st.session_state.training_data)
else:
    st.warning("No training data found. Please Upload the data first.")
