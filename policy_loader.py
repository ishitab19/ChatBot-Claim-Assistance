from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
print("ALL downloaded successfully!")
import faiss
import numpy as np
import pickle
import os
from pathlib import Path

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_vector_store(pdf_path):
    # Convert to absolute path if relative
    pdf_path = os.path.abspath(pdf_path)
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, "policy_index.faiss")

    with open("policy_texts.pkl", "wb") as f:
        pickle.dump(texts, f)

    print("Policy compressed & indexed successfully!")

if __name__ == "__main__":
    # Create policies directory if it doesn't exist
    policies_dir = "policies"
    os.makedirs(policies_dir, exist_ok=True)
    
    pdf_file = os.path.join(policies_dir, "sample_policy.pdf")
    
    if not os.path.exists(pdf_file):
        print(f"Error: {pdf_file} not found.")
        print(f"Please place your PDF file at: {os.path.abspath(pdf_file)}")
    else:
        build_vector_store(pdf_file)
