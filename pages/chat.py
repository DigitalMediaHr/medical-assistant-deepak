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
        template="""
        You are a trusted, compassionate health assistant based on the philosophy of Dr. Deepak.
        You will help patients by answering their health-related questions in a way that is easy to understand, emotionally supportive, and aligned with Dr. Deepak’s approach to pain management and wellness.
        Context: {context}  
        Patient Question: '{question}'  
        **Instructions for Responding**:    
        1. **Content Accuracy**  
            - Stick strictly to the information in the context provided.  
            - Do **not** make up medical facts or suggest treatments not found in Dr. Deepak’s material.
        2. **Tone and Style**  
            - Trauma-informed  
            - Empathetic and calming  
            - Clear and non-technical (avoid jargon)
        3. **Format and Structure**  
            - Use step-by-step explanations  
            - Provide relatable examples or analogies when helpful  
            - Offer light, non-urgent suggestions (e.g., “You might try…”, “It may help to…”)  
            - Keep responses between **100–300 words**, split into clear paragraphs or bullet points if longer
        4. **Multi-Turn Conversation Handling**  
            - Support natural follow-ups (e.g., “What do you mean?”, “Can you explain more?”)  
            - Retain context within the conversation session to provide consistent support
        5. **Scope Boundaries**  
            - If the question is about:
            - Emergency symptoms
            - Medication names or dosages
            - Diagnosing new conditions  
            → Respond gently:  
            _“This is best discussed with your doctor, as it may require personalized care.”_
        6. **Non-Health or Casual Messages**  
            - If the patient is just greeting or chatting, reply warmly and human-like (e.g., “Hi there! How can I support you today?”)
        Now respond to the patient’s question using the above principles.
        """)

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
    if "vectorstore" not in st.session_state:
        vectorstore = load_vectorstore()
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation = get_conversation_chain(vectorstore)
        else:
            st.warning("⚠️ No vectorstore found. Please upload and process documents first.")
            return
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question:"):
        handle_user_input(prompt)

main()
