import os
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY"))

while True:
    print("you:",end="",flush=True)
    user = input()
    
    if user.lower() == "exit" or user.lower() == "quit" or user.lower() == "close":
        break
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=user
        )

        embeddings = response.embeddings[0].values

        print(f"Input Text:{user} | embedding length:{len(embeddings)} | embedding:{embeddings}")

    except Exception as e:
        print(f"An error occurred: {e}")
        continue