# Wrapper for visual_eval compatibility - delegates to unified utils.llms module
from utils.llms.openai import openai_api_call as _openai_api_call
from .constant import DEFAULT_TEMPERATURE


def openai_api_call(
    api_key: str,
    diff: str,
    system_prompt: str,
    model: str = "gpt-4.1-2025-04-14",  # State of art openai model https://platform.openai.com/docs/models/gpt-4.1
    temperature: float = DEFAULT_TEMPERATURE,  # for greedy decoding to remove potential randomness
) -> str:
    """OpenAI API call wrapper for visual_eval compatibility."""
    return _openai_api_call(
        api_key=api_key,
        diff=diff,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
    )
