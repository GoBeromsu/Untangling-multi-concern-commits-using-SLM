"""OpenAI API handler for RQ0 experiment."""

import openai
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API configuration will be passed from config

# Structured output format for OpenAI
OPENAI_STRUCTURED_OUTPUT_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "commit_concerns_response",
        "schema": {
            "type": "object",
            "properties": {
                "concerns": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["concerns"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def load_openai_client(model_name: str) -> Dict[str, Any]:
    """Load OpenAI client and return model info."""
    return {
        "type": "openai",
        "model_name": model_name,
        "client": openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    }


def get_openai_prediction(
    model_info: Dict[str, Any], prompt: str, temperature: float, max_tokens: int
) -> str:
    """Get prediction from OpenAI API."""
    try:
        response = model_info["client"].chat.completions.create(
            model=model_info["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=OPENAI_STRUCTURED_OUTPUT_FORMAT,
        )
        prediction = response.choices[0].message.content or "No response from API."
    except openai.APIError as e:
        prediction = f"An OpenAI API error occurred: {e}"
    except Exception as e:
        prediction = f"An unexpected error occurred: {e}"

    return prediction
