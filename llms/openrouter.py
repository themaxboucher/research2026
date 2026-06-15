import os
from functools import lru_cache

from dotenv import load_dotenv
from openrouter import OpenRouter

load_dotenv()


@lru_cache(maxsize=1)
def _load_client() -> OpenRouter:
    return OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])


def get_completion(model_name: str, prompt: str) -> str:
    response = _load_client().chat.send(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
