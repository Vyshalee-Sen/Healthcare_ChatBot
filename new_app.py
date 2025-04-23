import streamlit as st
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="Healthcare Chatbot", layout="centered")

# --- Load Sentence-BERT Model and GPT2 Pipeline ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
llm = pipeline("text-generation", model="gpt2")

# --- Load and Chunk the Text File ---
@st.cache_resource
def load_chunks():
    with open('converted_text.txt', 'r', encoding='utf-8') as file:
        raw_text = file.read()

    cleaned_text = re.sub(r'\n+', '\n', raw_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    def chunk_text(text, chunk_size=500):
        words = text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    chunks = chunk_text(cleaned_text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return chunks, index

chunks, index = load_chunks()

# --- Streamlit UI Setup ---

st.title("📄 Healthcare ChatBot")

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Query Input
query = st.text_input("Ask a question:")

# Handle Query
if query:
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k=1)
    relevant_chunk = chunks[I[0][0]]

    prompt = f"Context: {relevant_chunk}\nQuestion: {query}\nAnswer:"
    llm_output = llm(prompt, max_new_tokens=50)[0]['generated_text']

    st.session_state.chat_history.append((query, llm_output))

# Display Chat History
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
