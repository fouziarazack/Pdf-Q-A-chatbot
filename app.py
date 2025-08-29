import streamlit as st
import faiss
import os
import tempfile
from PyPDF2 import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

# Load Hugging Face model for embeddings & QA
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
    return embedder, qa_model

embedder, qa_model = load_models()

# Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Create FAISS index
def create_faiss_index(text, embedder):
    # Split into chunks
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = embedder.encode(chunks)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks

# Retrieve relevant chunks
def search_pdf(query, index, chunks, embedder, top_k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in I[0]]

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="PDF Q/A Chatbot", layout="wide")
st.title("📄 PDF Question Answering Chatbot")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Save & process PDF
    text = extract_text_from_pdf(uploaded_file)
    index, chunks = create_faiss_index(text, embedder)

    st.success("✅ PDF uploaded and indexed. You can now ask questions!")

    question = st.text_input("Ask a question about your PDF:")

    if question:
        # Retrieve context
        retrieved_chunks = search_pdf(question, index, chunks, embedder, top_k=3)
        context = " ".join(retrieved_chunks)

        # Ask model
        input_text = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        response = qa_model(input_text, max_length=200, do_sample=False)

        st.markdown(f"🤖 **Answer:** {response[0]['generated_text']}")
