import os
import requests
from dotenv import load_dotenv

def test_fireworks():
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY not found in environment variables.")
        return

    # Fireworks AI chat completions endpoint
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "accounts/fireworks/models/deepseek-v4-pro",
        "messages": [
            {"role": "user", "content": "Say hello and confirm you are working"}
        ],
        "max_tokens": 100,
        "temperature": 0.0
    }
    
    print("Sending request to Fireworks AI...")
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print("\nSuccess! Connection confirmed.")
            print("Response from model:")
            print("-" * 40)
            print(message.strip())
            print("-" * 40)
        else:
            print(f"\nError: Request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"\nError: An exception occurred while making the request: {e}")

if __name__ == "__main__":
    test_fireworks()
