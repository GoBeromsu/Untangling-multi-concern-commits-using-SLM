"""LM Studio API handler for RQ0 experiments."""

import requests
import json
from typing import Dict, Any

# LM Studio configuration
DEFAULT_LMSTUDIO_URL = "http://localhost:1234"
CONNECTION_TIMEOUT_SECONDS = 60

# Response schema for commit classification
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["docs", "test", "cicd", "build", "refactor", "feat", "fix"],
            },
        },
        "count": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["types", "count", "reason"],
    "additionalProperties": False,
}

LMSTUDIO_STRUCTURED_OUTPUT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "commit_classification_response",
        "schema": RESPONSE_SCHEMA,
        "strict": True,
    },
}


def load_lmstudio_client(model_name: str) -> Dict[str, Any]:
    """Load LM Studio client configuration."""
    return {
        "type": "lmstudio",
        "model_name": model_name,
        "base_url": DEFAULT_LMSTUDIO_URL,
    }


def get_lmstudio_prediction(
    model_info: Dict[str, Any],
    user_prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Get prediction from LM Studio API."""
    try:
        url = f"{model_info['base_url']}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }

        data = {
            "model": model_info["model_name"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": LMSTUDIO_STRUCTURED_OUTPUT_FORMAT,
        }

        response = requests.post(
            url, headers=headers, json=data, timeout=CONNECTION_TIMEOUT_SECONDS
        )
        response.raise_for_status()

        response_json = response.json()
        response_content = response_json["choices"][0]["message"]["content"]

        return response_content

    except requests.exceptions.Timeout as e:
        return f"An error occurred while calling LM Studio API: Request timed out after {CONNECTION_TIMEOUT_SECONDS}s - {e}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred while calling LM Studio API: {e}"
    except KeyError as e:
        return f"An error occurred while parsing LM Studio response: Missing key {e}"
    except Exception as e:
        return f"An error occurred while processing the response: {e}"
