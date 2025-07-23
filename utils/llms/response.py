"""Response parsing utilities for LLM outputs (visual_eval standard)."""

import json
from typing import List, Set


def parse_model_response(model_response: str) -> List[str]:
    """
    Parse model response with consistent error handling (visual_eval standard).

    Args:
        model_response: JSON string from model API

    Returns:
        List of concern types
    """
    prediction_data = json.loads(model_response)
    predicted_concern_types = prediction_data.get("types", [])
    return predicted_concern_types


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
