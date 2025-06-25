# src/llms/openai.py
import os
import openai
from typing import Optional, List
from pydantic import BaseModel


def initialize_client(api_key: Optional[str] = None) -> openai.OpenAI:
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY "
            "environment variable or pass it to the constructor."
        )
    return openai.OpenAI(api_key=resolved_api_key)


def get_completion(
    client: openai.OpenAI,
    prompt: str,
    model: str = "gpt-4.1-2025-04-14",  # State of art openai model https://platform.openai.com/docs/models/gpt-4.1
    temperature: float = 0.0,  # for greedy decoding to remove potential randomness
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                # Using user role only following https://github.com/0x404/conventional-commit-classification methodology
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or "No response from API."
    except openai.APIError as e:
        return f"An OpenAI API error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# Structured output models for diff analysis
class CommitType(BaseModel):
    type: str
    confidence: float


class DiffAnalysisResult(BaseModel):
    types: List[CommitType]
    count: int


def analyze_diff_structured(
    diff_content: str,
    prompt_template: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-2024-08-06",  # Model that supports structured output
    temperature: float = 0.0,
) -> DiffAnalysisResult:
    # Initialize client inline as requested
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY "
            "environment variable or pass it to the constructor."
        )

    client = openai.OpenAI(api_key=resolved_api_key)

    # Format the prompt with diff content
    formatted_prompt = prompt_template.format(diff=diff_content)

    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "user", "content": formatted_prompt},
            ],
            response_format=DiffAnalysisResult,
            temperature=temperature,
        )

        parsed_result = completion.choices[0].message.parsed
        if parsed_result:
            return parsed_result
        else:
            # Fallback if parsing fails
            raise Exception(
                f"Parsing failed. Refusal: {completion.choices[0].message.refusal}"
            )

    except openai.APIError as e:
        raise Exception(f"OpenAI API error occurred: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")
