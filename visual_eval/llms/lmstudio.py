# Wrapper for visual_eval compatibility - delegates to unified utils.llms module
from typing import List, Tuple, Dict, Any
from utils.llms.lmstudio import (
    get_models as _get_models,
    load_model as _load_model,
    api_call as _api_call,
    clear_cache as _clear_cache,
)
from .constant import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


def get_models() -> Tuple[List[str], str]:
    """Get available LLM models from LM Studio (visual_eval compatibility wrapper)."""
    return _get_models()


def load_model(model_name: str) -> Any:
    """Load model with optimized configuration (visual_eval compatibility wrapper)."""
    return _load_model(model_name)


def api_call(
    model_name: str,
    diff: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """LM Studio API call wrapper for visual_eval compatibility."""
    return _api_call(
        model_name=model_name,
        commit=diff,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def clear_cache() -> None:
    """Clear the model cache (visual_eval compatibility wrapper)."""
    return _clear_cache()
