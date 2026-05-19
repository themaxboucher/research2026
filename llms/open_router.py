from openrouter import OpenRouter
import os

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

openrouter = OpenRouter(api_key=OPENROUTER_API_KEY)

def get_completion(model: str, prompt: str) -> str:
    response = openrouter.chat.send(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print(get_completion("meta-llama/llama-3.1-8b-instruct", "Hello, world!"))