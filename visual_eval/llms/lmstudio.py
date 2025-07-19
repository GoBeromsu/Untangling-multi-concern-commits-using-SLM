import requests
import json
from typing import Any, Dict, List, Tuple

from .constant import (
    LMSTUDIO_STRUCTURED_OUTPUT_FORMAT,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    CONNECTION_TIMEOUT_SECONDS,
)


def check_lmstudio_connection(base_url: str = DEFAULT_LMSTUDIO_URL) -> bool:
    """
    Check if LM Studio API is accessible.

    Args:
        base_url: Base URL for LM Studio API

    Returns:
        True if connection is successful, False otherwise
    """
    try:
        url = f"{base_url}/v1/models"
        response = requests.get(url, timeout=CONNECTION_TIMEOUT_SECONDS)
        response.raise_for_status()
        return True
    except Exception:
        return False


def get_lmstudio_models(
    base_url: str = DEFAULT_LMSTUDIO_URL,
) -> Tuple[List[str], str]:
    """
    Get available models from LM Studio API.

    Args:
        base_url: Base URL for LM Studio API

    Returns:
        Tuple of (model_names_list, error_message)
        If successful, error_message is empty
    """
    try:
        url = f"{base_url}/v1/models"
        response = requests.get(url, timeout=CONNECTION_TIMEOUT_SECONDS)
        response.raise_for_status()

        models_data = response.json()
        model_names = [model["id"] for model in models_data.get("data", [])]

        return model_names, ""
    except requests.exceptions.RequestException as e:
        return [], f"Request error: {e}"
    except Exception as e:
        return [], f"Error: {e}"


def lmstudio_api_call(
    model_name: str,
    diff: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: str = DEFAULT_LMSTUDIO_URL,
) -> str:
    """
    Call LM Studio API for commit classification using HTTP requests.

    Args:
        model_name: Name of the model to use (e.g., "codellama-7b@f16")
        diff: Git diff content to analyze
        system_prompt: System prompt for the model
        temperature: Sampling temperature (0.0 for greedy decoding)
        max_tokens: Maximum tokens to generate (use -1 for unlimited)
        base_url: Base URL for LM Studio API

    Returns:
        JSON string containing the structured response
    """
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": diff},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": LMSTUDIO_STRUCTURED_OUTPUT_FORMAT,
    }

    # Try up to 2 times for potential model warm-up
    for attempt in range(2):
        try:
            response = requests.post(
                url, headers=headers, json=data, timeout=CONNECTION_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            if attempt == 0:
                continue
            return f"An error occurred while calling LM Studio API: {e}"
        except Exception as e:
            return f"An error occurred while processing the response: {e}"
