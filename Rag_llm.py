# your_module.py
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login



# Load pre-trained models and index once
def load_model():
    login(token="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")
    # Sentence-BERT for embedding
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # GPT-Neo for generation
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    gpt_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    generator = pipeline('text-generation', model=gpt_model, tokenizer=tokenizer)

    # Load or recreate FAISS index and chunks
    with open('converted_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)

    chunks = [' '.join(text.split()[i:i+500]) for i in range(0, len(text.split()), 500)]
    embeddings = embedder.encode(chunks)
    embedding_matrix = np.array(embeddings)

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    return generator, embedder, index, chunks

# Retrieve context chunks
def retrieve_chunks(query, embedder, index, chunks, top_k=1):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

# Generate answer using GPT-Neo
def get_response(query, generator, embedder, index, chunks):
    context = ' '.join(retrieve_chunks(query, embedder, index, chunks))
    input_text = f"{context}\n\nQuestion: {query}\nAnswer:"
    output = generator(input_text, max_new_tokens=50, num_return_sequences=1)
    return output[0]['generated_text']
