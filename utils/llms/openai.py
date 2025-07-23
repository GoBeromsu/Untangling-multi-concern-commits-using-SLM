"""Unified OpenAI API utilities for all experiments."""

import openai
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

from .constant import (
    OPENAI_STRUCTURED_OUTPUT_FORMAT,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)

load_dotenv()


def openai_api_call(
    api_key: str,
    diff: str,
    system_prompt: str,
    model: str = "gpt-4-1106-preview",  # State of art OpenAI model
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """
    OpenAI API call (visual_eval standard).

    Args:
        api_key: OpenAI API key
        diff: Code diff content
        system_prompt: System prompt for the model
        model: OpenAI model name
        temperature: Sampling temperature

    Returns:
        JSON string containing the model response
    """
    client = openai.OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": diff},
            ],
            temperature=temperature,
            response_format=OPENAI_STRUCTURED_OUTPUT_FORMAT,
        )
        return response.choices[0].message.content or "No response from API."
    except openai.APIError as e:
        return f"An OpenAI API error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def load_openai_client(model_name: str) -> Dict[str, Any]:
    """
    Load OpenAI client and return model info (utils style).

    Args:
        model_name: Name of the OpenAI model

    Returns:
        Dict containing client info for structured API calls
    """
    return {
        "type": "openai",
        "model_name": model_name,
        "client": openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    }


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
