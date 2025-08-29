from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=800, chunk_overlap=120):
    """
    Splits text into overlapping chunks for better retrieval.
    chunk_size: how many characters per chunk
    chunk_overlap: how many characters each chunk overlaps
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start = max(0, end - chunk_overlap)
    
    return chunks


from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert chunks into embeddings
def get_embeddings(chunks):
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


import chromadb
from chromadb.utils import embedding_functions

def store_embeddings(chunks, embeddings):
    # Create a Chroma client (saves to local folder "chroma_db")
    client = chromadb.PersistentClient(path="chroma_db")

    # Create a collection (like a table in DB)
    collection = client.get_or_create_collection(
        name="pdf_chunks",
        embedding_function=None
    )

    # Add chunks + embeddings
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"chunk_{i}"],          # unique ID
            documents=[chunk],           # store original text
            embeddings=[emb.tolist()]    # convert numpy array → list
        )

    print("✅ Chunks stored in ChromaDB")

# Test it
if __name__ == "__main__":
    pdf_path = "UNIT-4.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    store_embeddings(chunks, embeddings)
    print("✅ PDF processed and stored in ChromaDB")




