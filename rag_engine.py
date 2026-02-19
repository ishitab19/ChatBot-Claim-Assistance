import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

# Lazy load index and texts
index = None
texts = None

def load_policy_data():
    global index, texts
    if index is None or texts is None:
        try:
            index = faiss.read_index("policy_index.faiss")
            with open("policy_texts.pkl", "rb") as f:
                texts = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Policy files not found: {e}")

def retrieve_policy(query, top_k=3):
    load_policy_data()
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = [texts[i] for i in indices[0]]
    return "\n".join(results)

def generate_answer(query):
    context = retrieve_policy(query)

    prompt = f"""
    You are an insurance assistant.
    Use the policy context to answer accurately.

    Policy Context:
    {context}

    User Question:
    {query}
    """

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )

    return response.choices[0].message.content
