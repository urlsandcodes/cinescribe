import os
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_ID = os.getenv("OPENROUTER_MODEL_ID", "google/gemma-4-31b-it:free")

print(f"Testing OpenRouter API")
print(f"Model ID: {OPENROUTER_MODEL_ID}")
print(f"Key length: {len(OPENROUTER_API_KEY) if OPENROUTER_API_KEY else 0}")

if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY is not set in your .env file!")
    exit(1)

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}
payload = {
    "model": OPENROUTER_MODEL_ID,
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Say hello and confirm your model name."}
    ],
    "temperature": 0.7,
    "max_tokens": 100
}

try:
    print("Sending request to OpenRouter...")
    resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
    print(f"Response Status: {resp.status_code}")
    if resp.status_code == 200:
        content = resp.json()["choices"][0]["message"]["content"]
        print("Success! Response:")
        print(content)
    else:
        print(f"Failed: {resp.text}")
except Exception as e:
    print(f"Exception raised: {e}")
