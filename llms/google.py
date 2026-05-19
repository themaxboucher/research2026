import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

google = genai.Client(api_key=GOOGLE_API_KEY)

def get_completion(model: str, prompt: str) -> str:
    response = google.models.generate_content(model=model, contents=prompt)
    return response.text