"""Response parsing utilities for LLM outputs (visual_eval standard)."""

import json
from typing import List, Tuple, Set


def parse_model_response(model_response: str) -> Tuple[List[str], str]:
    """
    Parse model response with consistent error handling (visual_eval standard).

    Args:
        model_response: JSON string from model API

    Returns:
        Tuple of (concern_types, reasoning)
    """
    try:
        prediction_data = json.loads(model_response)
        predicted_concern_types = prediction_data.get("types", [])
        model_reasoning = prediction_data.get("reason", "No reasoning provided")
        return predicted_concern_types, model_reasoning
    except json.JSONDecodeError as e:
        return (
            [],
            f"JSON Parse Error: {str(e)}. Raw response: {model_response[:200]}...",
        )
    except Exception as e:
        return [], f"Parse Error: {str(e)}. Raw response: {model_response[:200]}..."


def parse_prediction_to_set(prediction: str) -> Set[str]:
    """
    Parse prediction to set of concern types for RQ experiment compatibility.

    Args:
        prediction: JSON string from model API

    Returns:
        Set of predicted concern types
    """
    try:
        output_json = json.loads(prediction)
        predicted_concerns = set(output_json.get("types", []))
        return predicted_concerns
    except (json.JSONDecodeError, KeyError):
        return set()


def parse_ground_truth_to_set(types_json: str) -> Set[str]:
    """
    Parse ground truth types to set for RQ experiment compatibility.

    Args:
        types_json: JSON string containing ground truth types

    Returns:
        Set of ground truth concern types
    """
    try:
        return set(json.loads(types_json))
    except (json.JSONDecodeError, TypeError):
        return set()
