from dotenv import load_dotenv
import os
from query_pdf import search_pdf
from transformers import pipeline

# Load Hugging Face model (Flan-T5 for Q/A)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def ask_chatbot(question):
    # Step 1: Retrieve relevant chunks from PDF
    results = search_pdf(question, top_k=3)
    retrieved_chunks = " ".join(results["documents"][0])

    # Step 2: Ask Hugging Face model
    prompt = f"Context: {retrieved_chunks}\n\nQuestion: {question}\nAnswer:"

    response = qa_pipeline(prompt, max_length=200, do_sample=False)
    return response[0]["generated_text"]

# Test loop
if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break
        answer = ask_chatbot(question)
        print("\n🤖 Chatbot:", answer)


