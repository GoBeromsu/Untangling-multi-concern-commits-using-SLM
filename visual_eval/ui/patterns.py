"""
Common patterns and utilities following frontend design principles.
"""

import json
import streamlit as st
from typing import Dict, Any, Tuple, List, Optional


def parse_model_response(model_response: str) -> Tuple[List[str], str]:
    """
    Parse model response with consistent error handling.

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
    except json.JSONDecodeError:
        return [], "Failed to parse model response"


def extract_test_case_data(
    test_case: Dict[str, Any]
) -> Tuple[str, List[str], List[str]]:
    """
    Extract data from test case with consistent pattern.

    Args:
        test_case: Test case data dictionary

    Returns:
        Tuple of (diff, actual_concern_types, shas)
    """
    diff = test_case.get("tangleChange", "")
    atomic_changes = test_case.get("atomicChanges", [])
    actual_concern_types = [change.get("label", "") for change in atomic_changes]
    shas = test_case.get("shas", [])
    return diff, actual_concern_types, shas
