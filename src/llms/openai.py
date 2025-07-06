import openai
from typing import Any, Dict

from llms.constant import OPENAI_STRUCTURED_OUTPUT_FORMAT, DEFAULT_TEMPERATURE


def openai_api_call(
    api_key: str,
    diff: str,
    system_prompt: str,
    model: str = "gpt-4.1-2025-04-14",  # State of art openai model https://platform.openai.com/docs/models/gpt-4.1
    temperature: float = DEFAULT_TEMPERATURE,  # for greedy decoding to remove potential randomness
) -> str:
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
