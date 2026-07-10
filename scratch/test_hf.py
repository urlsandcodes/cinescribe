import os
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "google/gemma-4-31B-it")

print(f"Testing Hugging Face Serverless API")
print(f"Model ID: {HF_MODEL_ID}")
print(f"Token length: {len(HF_TOKEN) if HF_TOKEN else 0}")

if not HF_TOKEN:
    print("Error: HF_TOKEN is not set in your .env file!")
    exit(1)

url = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}
payload = {
    "model": HF_MODEL_ID,
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Say hello and confirm your model name."}
    ],
    "temperature": 0.7,
    "max_tokens": 100
}

try:
    print("Sending request to Hugging Face...")
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
