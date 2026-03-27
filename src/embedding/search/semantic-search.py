import os
import google.genai as genai
from dotenv import load_dotenv
import numpy as np
import pickle
from pathlib import Path
from google.genai import types
from time import sleep

load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY"))

max_tokens = int(os.getenv("Max_Tokens").strip() or 8000)

max_history_length = int(os.getenv("Max_History_Length").strip() or 6)

cache_file = "embeddings.pkl"
cache_dir = Path.cwd()/"data"

gen_model_name = ""
embed_model_name = ""

cache_file_path = cache_dir/cache_file

system_prompt = """
    Rules:
    - Answer in friednly and easy to understand terms
    - Keep the answer short an compact
    - dont ask follow up questions
    - Explain step by step
    - Use only provided context
    - Do not hallucinate
    - If you don't know the answer, say "I don't know" instead of making up an answer.
    - Never share any personal information or sensitive data.
    """

if gen_model_name == "":
    gen_model_name = os.getenv("GEN_MODEL_NAME").strip() or client.models.list()[0].name

documents = [
    "Flutter is used for mobile development",
    "Python is used for AI",
    "Dart is Flutter's language",
    "AI uses machine learning"
]

if embed_model_name == "":
    embed_model_name = os.getenv("EMBEDDING_MODEL_NAME").strip() or "gemini-embedding-001"

history = []

def estimate_tokens(text):
    return int(len(text) / 4)  # Rough estimate: 1 token ≈ 4 characters

def count_history_tokens(history, system_prompt):
    total_tokens = estimate_tokens(system_prompt)
    for entry in history:
        for part in entry["parts"]:
            total_tokens += estimate_tokens(part["text"])
    return total_tokens

def trim_history(history, system_prompt):
    while count_history_tokens(history, system_prompt) > max_tokens:
        if len(history) >= 2:
            history.pop(0) # Remove the oldest User entry
            history.pop(0) # Remove the corresponding Model entry
        else:
            break
    return history

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

        best_match = None
        best_score = -1
        scores = []
        
        for doc, emb in doc_embedding:
            score = cosine_similarity(qry_emb, emb)
            scores.append((doc, score))
        
        scores.sort(key=lambda x :x[1], reverse=True)

        top_k = scores[:3]

        context = "\n".join([doc for doc, _ in top_k])

        bot_reply = ""

        prompt = f'''
            answer the question only using the context below
            context:{context}
            question:{qry}
        '''
        history.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        # Trim history BEFORE sending    
        history = trim_history(history, system_prompt)

        stream = client.models.generate_content_stream(
            model=gen_model_name,
            contents=history, # Pass the conversation history to maintain context
            config=types.GenerateContentConfig(
                temperature=0.7, # Adjust the temperature for more creative responses
                system_instruction=[{"text": system_prompt}]
                # Provide the system prompt as a list to ensure it's included in the generation process
            ),
        )

        print(f"{gen_model_name}: ", end="", flush=True)

        for chunk in stream:
            if chunk.text:
                for line in chunk.text.splitlines():
                    for char in line:
                        print(char, end="", flush=True)
                        sleep(0.01)
                        bot_reply += char
                    print()  # Move to the next line after the bot finishes replying
                    bot_reply += "\n"  # Add a newline after each line of the bot's response


    except Exception as e:
        print(f"An error occurred: {e}")
        continue

    history.append({
        "role": "model",
        "parts": [{"text": bot_reply}]
    })