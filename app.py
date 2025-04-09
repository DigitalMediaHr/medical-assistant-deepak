import streamlit as st

pg = st.navigation([st.Page("pages/data_Preprocessing.py", title="Upload KnowledgeBase (Data)"),st.Page("pages/training_data.py", title="Training data"),st.Page("pages/chat.py", title="Start Chat")])
pg.run()