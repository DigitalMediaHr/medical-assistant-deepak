import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
load_dotenv()

VECTORSTORE_PATH = "vectorstore.index"

def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        try:
            embeddings = OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-large"
            )
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception as e:
            st.error(f"Error loading vectorstore: {e}")
    return None

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You have access to the following context: {context}.  
                    Use this information to answer the question: '{question}'.  
                    - Maintain the same style, tone, and explanation style as in the context.  
                    - Stick strictly to the details provided—do not add anything extra.  
                    - If the context does not contain the answer and the question is unrelated to the context, respond naturally instead of saying you cannot help.  
                    - If the input is just a greeting or small talk, reply in a friendly and human-like way."""
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

def handle_user_input(user_question):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation({'question': user_question})
                bot_response = response['chat_history'][-1].content
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                st.markdown(bot_response)
            except Exception as e:
                st.error(f"Error generating response: {e}")

def main():
    st.set_page_config(page_title="Prof. Dr Deepak Ravindran", layout="wide")
    st.title("Prof. Dr Deepak Ravindran")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_chain" not in st.session_state:
        vectorstore = load_vectorstore()
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation_chain = get_conversation_chain(vectorstore)
        else:
            st.warning("⚠️ No vectorstore found. Please upload and process documents first.")
            return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question:"):
        handle_user_input(prompt)

main()
