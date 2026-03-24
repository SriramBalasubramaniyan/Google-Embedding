import os
import google.genai as genai
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY"))

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

while True:
    print()
    print("Text 1:",end="",flush=True)
    txt1 = input().lower().strip()

    print("Text 2:",end="",flush=True)
    txt2 = input().lower().strip()
    
    try:
        emb1 = client.models.embed_content(
            model="gemini-embedding-001",
            contents=txt1
        ).embeddings[0].values

        emb2 = client.models.embed_content(
            model="gemini-embedding-001",
            contents=txt2
        ).embeddings[0].values

        score = cosine_similarity(emb1, emb2)
        print()
        print(f"{txt1}|{txt2}\nSimilarity:{score}")

    except Exception as e:
        print(f"An error occurred: {e}")
        continue