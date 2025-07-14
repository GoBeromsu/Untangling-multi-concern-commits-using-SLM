"""OpenAI API handler for RQ0 experiment."""

import openai
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# Response schema matching LM Studio format
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

OPENAI_STRUCTURED_OUTPUT_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "commit_classification_response",
        "schema": RESPONSE_SCHEMA,
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
    model_info: Dict[str, Any],
    user_prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Get prediction from OpenAI API."""
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
