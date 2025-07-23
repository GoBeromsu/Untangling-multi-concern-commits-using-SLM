# LLM handlers package for common use across all RQ experiments

from .openai import openai_api_call, load_openai_client, get_openai_prediction
from .lmstudio import (
    get_models,
    load_model,
    api_call,
    clear_cache,
)
from .constant import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    COMMIT_TYPES,
    RESPONSE_SCHEMA,
    OPENAI_STRUCTURED_OUTPUT_FORMAT,
    LMSTUDIO_STRUCTURED_OUTPUT_FORMAT,
)

__all__ = [
    "openai_api_call",
    "load_openai_client",
    "get_openai_prediction",
    "get_models",
    "load_model",
    "api_call",
    "clear_cache",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "COMMIT_TYPES",
    "RESPONSE_SCHEMA",
    "OPENAI_STRUCTURED_OUTPUT_FORMAT",
    "LMSTUDIO_STRUCTURED_OUTPUT_FORMAT",
]
