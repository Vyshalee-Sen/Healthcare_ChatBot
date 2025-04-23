# app.py
import streamlit as st
from Rag_llm import load_model, get_response  # You need to define these based on your notebook
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


# Load model and retriever (do it only once)
@st.cache_resource
def setup():
    model, retriever = load_model()
    return model, retriever

model, retriever = setup()

# UI
st.title("RAG-based Question Answering App")

user_query = st.text_input("Ask your question:")

if user_query:
    with st.spinner("Fetching answer..."):
        response = get_response(user_query, model, retriever)
        st.success(response)
