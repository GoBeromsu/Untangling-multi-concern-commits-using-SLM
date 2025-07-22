"""Unified LM Studio API utilities for all experiments."""

from typing import List, Tuple, Dict, Any
import time
import lmstudio as lms

from .constant import (
    RESPONSE_SCHEMA,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    LMSTUDIO_MODEL_CONFIG,
)

# Global state for loaded models
loaded_models: Dict[str, Any] = {}


def get_models() -> Tuple[List[str], str]:
    """
    Get available LLM models from LM Studio.

    Returns:
        Tuple of (model_names_list, error_message)
    """
    try:
        downloaded = lms.list_downloaded_models("llm")
        model_names = [model.model_key for model in downloaded]
        return model_names, ""
    except Exception as e:
        return [], f"Error: {e}"


def load_model(model_name: str) -> Any:
    """
    Load model with optimized configuration.

    Args:
        model_name: Name of the model to load

    Returns:
        Loaded model instance
    """
    if model_name not in loaded_models:
        try:
            model = lms.llm(model_name, config=LMSTUDIO_MODEL_CONFIG)
            loaded_models[model_name] = model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    return loaded_models[model_name]


def api_call(
    model_name: str,
    diff: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Tuple[str, float]:
    """
    Call LM Studio API for commit classification with inference time measurement.

    Args:
        model_name: Name of the model to use
        diff: Git diff content to analyze
        system_prompt: System prompt for the model
        temperature: Sampling temperature (0.0 for greedy decoding)
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple of (JSON string containing structured response, inference_time_in_seconds)
    """
    start_time = time.time()

    try:
        # Load model when actually needed
        model = load_model(model_name)

        # Prepare messages in correct format for LM Studio
        messages = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": diff},
            ]
        }

        # Use model.respond() with proper message format and configuration
        response = model.respond(
            messages,
            config={
                "temperature": temperature,
                "maxTokens": max_tokens,
                "structured": {
                    "type": "json",
                    "jsonSchema": RESPONSE_SCHEMA,
                },
            },
        )

        inference_time = time.time() - start_time

        # Handle response based on LM Studio SDK format
        if hasattr(response, "parsed") and response.parsed:
            # For structured response, return the parsed JSON as string
            import json

            return json.dumps(response.parsed), inference_time
        elif hasattr(response, "content") and response.content:
            # For regular response, return content directly
            return response.content, inference_time
        else:
            return "No response from model.", inference_time

    except Exception as e:
        inference_time = time.time() - start_time
        return f"An error occurred while calling LM Studio: {e}", inference_time


def clear_cache() -> None:
    """Clear the model cache."""
    global loaded_models
    loaded_models.clear()
