import chromadb
from sentence_transformers import SentenceTransformer

# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_pdf(question, top_k=3):
    # Connect to the ChromaDB where we stored embeddings
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("pdf_chunks")

    # Convert user question → embedding
    query_embedding = model.encode([question], normalize_embeddings=True)

    # Search top_k similar chunks
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    return results