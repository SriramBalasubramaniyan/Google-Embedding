import os
import google.genai as genai
from dotenv import load_dotenv
import numpy as np
import pickle
from pathlib import Path

load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY"))

max_tokens = int(os.getenv("Max_Tokens").strip() or 8000)

max_history_length = int(os.getenv("Max_History_Length").strip() or 6)

cache_file = "embeddings.pkl"
cache_dir = Path.cwd()/"data"

embed_model_name = ""

cache_file_path = cache_dir/cache_file

documents = []

if embed_model_name == "":
    embed_model_name = os.getenv("EMBEDDING_MODEL_NAME").strip() or "gemini-embedding-001"

try:
    cache_dir.mkdir(parents=True,exist_ok=True)
except OSError as e:
    print(e)

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

if os.path.exists(cache_file_path):
    with open(cache_file_path, "rb") as f:
        doc_embedding = pickle.load(f)
else:
    doc_embedding = []
    for doc in documents:
        emb = client.models.embed_content(
                model=embed_model_name,
                contents=doc                
            ).embeddings[0].values
        
        doc_embedding.append((doc,emb))

    with open(cache_file_path, "wb") as f:
        pickle.dump(doc_embedding, f)

while True:
    print()
    print("Query:",end="",flush=True)
    qry = input().lower().strip()
    
    if qry.lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break

    try:
        qry_emb = client.models.embed_content(
            model=embed_model_name,
            contents=qry
        ).embeddings[0].values

        scores = []
        
        for doc, emb in doc_embedding:
            score = cosine_similarity(qry_emb, emb)
            scores.append((doc, score))
        
        top_k = [(doc, score) for doc, score in scores if score > 0.6][:3]

        if not top_k:
            print("No relevant context found")
            continue

        context = "\n".join([doc for doc, _ in top_k])

        print(f"Best Match:\n{context}")
    except Exception as e:
        print(f"An error occurred: {e}")
        continue
