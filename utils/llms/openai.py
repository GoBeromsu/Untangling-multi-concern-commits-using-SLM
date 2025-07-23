"""Unified OpenAI API utilities for all experiments."""

import openai
import json
from typing import Dict, Any, List

from .constant import (
    OPENAI_STRUCTURED_OUTPUT_FORMAT,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)


def api_call(
    api_key: str,
    commit: str,
    system_prompt: str,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[str]:
    """
    Call OpenAI API with cached client.

    Args:
        api_key: OpenAI API key
        commit: Commit content to analyze
        system_prompt: System prompt for the model
        model: OpenAI model name
        temperature: Sampling temperature

    Returns:
        List of concern types
    """
    client = openai.OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": commit},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=OPENAI_STRUCTURED_OUTPUT_FORMAT,
        )
        print(f"Response: {response.choices[0].message.content}")
        response_json = response.choices[0].message.content or "{'types': []}"
        response_data = json.loads(response_json)
        return response_data.get("types", [])
    except openai.APIError as e:
        raise RuntimeError(f"OpenAI API error: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON response: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def get_openai_prediction(
    model_info: Dict[str, Any],
    user_prompt: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Get prediction from OpenAI API using structured client (visual_eval standard).

    Args:
        model_info: Model info dict from load_openai_client
        user_prompt: User prompt content
        system_prompt: System prompt for the model
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        JSON string containing the model response
    """
    client = model_info["client"]
    model = model_info["model_name"]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=OPENAI_STRUCTURED_OUTPUT_FORMAT,
        )
        return response.choices[0].message.content or "No response from API."

    except openai.APIError as e:
        return f"An OpenAI API error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
