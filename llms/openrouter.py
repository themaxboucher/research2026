import logging
import os
import random
import time
from functools import lru_cache

import httpx
from dotenv import load_dotenv

load_dotenv()

MAX_ATTEMPTS = 5
INITIAL_RETRY_DELAY_SECONDS = 2.0


@lru_cache(maxsize=1)
def _retryable_errors() -> tuple[type[Exception], ...]:
    # Imported lazily so this module and everything that imports it loads on hosts 
    # without the `openrouter` package installed.
    from openrouter import errors

    return (
        errors.TooManyRequestsResponseError,
        errors.ProviderOverloadedResponseError,
        errors.InternalServerResponseError,
        errors.BadGatewayResponseError,
        errors.ServiceUnavailableResponseError,
        errors.RequestTimeoutResponseError,
        errors.EdgeNetworkTimeoutResponseError,
        errors.NoResponseError,
        httpx.HTTPError,
    )


@lru_cache(maxsize=1)
def _load_client():
    from openrouter import OpenRouter

    return OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])


def get_completion(model_name: str, prompt: str) -> str:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = _load_client().chat.send(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except _retryable_errors() as error:
            if attempt == MAX_ATTEMPTS:
                raise
            # Full jitter so concurrent workers don't retry in lockstep.
            delay = INITIAL_RETRY_DELAY_SECONDS * 2 ** (attempt - 1) * random.random()
            logging.warning(
                "OpenRouter call to %s failed (attempt %d/%d), retrying in %.1fs: %s",
                model_name,
                attempt,
                MAX_ATTEMPTS,
                delay,
                error,
            )
            time.sleep(delay)
