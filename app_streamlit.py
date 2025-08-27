import streamlit as st
import requests

st.title("Local RAG Demo")
q = st.text_input("Ask a question about the docs:")
if st.button("Search"):
    resp = requests.post("http://localhost:8000/query", json={"q": q}).json()
    st.write("**Answer:**", resp["answer"])
    st.write("**Sources:**", resp["context_files"])
