"""Session state management utilities for consistent state handling."""

import streamlit as st
import pandas as pd
from typing import Optional, List, Literal

# Session state keys
API_PROVIDER_KEY = "selected_api"
MODEL_NAME_KEY = "selected_model"
AVAILABLE_MODELS_KEY = "lmstudio_available_models"
EVALUATION_RESULTS_KEY = "final_evaluation_results"

# Type definitions
ApiProvider = Literal["openai", "lmstudio"]


def get_api_provider() -> ApiProvider:
    """Get currently selected API provider with type safety."""
    return st.session_state.get(API_PROVIDER_KEY, "openai")


def set_api_provider(provider: ApiProvider, model_name: Optional[str] = None) -> None:
    """Set API provider and associated model with type safety."""
    st.session_state[API_PROVIDER_KEY] = provider
    st.session_state[MODEL_NAME_KEY] = model_name


def get_model_name() -> str:
    """Get currently selected model name."""
    return st.session_state.get(MODEL_NAME_KEY, "")


def get_available_models() -> List[str]:
    """Get list of available LM Studio models."""
    return st.session_state.get(AVAILABLE_MODELS_KEY, [])


def set_available_models(models: List[str]) -> None:
    """Set list of available LM Studio models."""
    st.session_state[AVAILABLE_MODELS_KEY] = models


def get_evaluation_results() -> Optional[pd.DataFrame]:
    """Get final evaluation results from session."""
    return st.session_state.get(EVALUATION_RESULTS_KEY)


def set_evaluation_results(results_df: pd.DataFrame) -> None:
    """Store final evaluation results in session."""
    st.session_state[EVALUATION_RESULTS_KEY] = results_df


def clear_evaluation_results() -> None:
    """Clear stored evaluation results."""
    if EVALUATION_RESULTS_KEY in st.session_state:
        del st.session_state[EVALUATION_RESULTS_KEY]
