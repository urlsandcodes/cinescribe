import os
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemma-4-31b-it")

print(f"Testing Google AI Studio OpenAI-compatible API")
print(f"Model ID: {GEMINI_MODEL_ID}")
print(f"Key length: {len(GEMINI_API_KEY) if GEMINI_API_KEY else 0}")

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY is not set in your .env file!")
    exit(1)

url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
headers = {
    "Authorization": f"Bearer {GEMINI_API_KEY}",
    "Content-Type": "application/json"
}
payload = {
    "model": GEMINI_MODEL_ID,
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Say hello and confirm your model name."}
    ],
    "temperature": 0.7,
    "max_tokens": 100
}

try:
    print("Sending request to Google AI Studio...")
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
