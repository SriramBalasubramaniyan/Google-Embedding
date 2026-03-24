import os
import google.genai as genai
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY"))

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

documents = [
    "Flutter is used for mobile development",
    "Python is used for AI",
    "Dart is Flutter's language",
    "AI uses machine learning"
]

doc_embedding = []

for doc in documents:
    emb = client.models.embed_content(
            model="gemini-embedding-001",
            contents=doc
            
        ).embeddings[0].values
    
    doc_embedding.append((doc,emb))

while True:
    print()
    print("Query:",end="",flush=True)
    qry = input().lower().strip()
    
    try:
        qry_emb = client.models.embed_content(
            model="gemini-embedding-001",
            contents=qry
        ).embeddings[0].values

        best_match = None
        best_score = -1
        
        for doc, emb in doc_embedding:
            score = cosine_similarity(qry_emb, emb)

            if score > best_score:
                best_score = score
                best_match = doc
        
        print(best_match)
    except Exception as e:
        print(f"An error occurred: {e}")
        continue