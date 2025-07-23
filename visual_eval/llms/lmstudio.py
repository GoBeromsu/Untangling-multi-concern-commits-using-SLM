# Wrapper for visual_eval compatibility - delegates to unified utils.llms module
from typing import List, Tuple, Dict, Any
from utils.llms.lmstudio import (
    get_models as _get_models,
    load_model as _load_model,
    clear_cache as _clear_cache,
)


def get_models() -> Tuple[List[str], str]:
    """Get available LLM models from LM Studio (visual_eval compatibility wrapper)."""
    return _get_models()


def load_model(model_name: str) -> Any:
    """Load model with optimized configuration (visual_eval compatibility wrapper)."""
    return _load_model(model_name)


def clear_cache() -> None:
    """Clear the model cache (visual_eval compatibility wrapper)."""
    return _clear_cache()
