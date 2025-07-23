"""API setup and configuration utilities for sidebar management."""

import os
import streamlit as st
from utils.llms.lmstudio import get_models
from utils.llms.constant import DEFAULT_OPENAI_MODEL
from .session import (
    set_api_provider,
    set_available_models,
    has_available_models,
    get_available_models,
    get_model_name,
)


def setup_openai_api() -> bool:
    """
    Setup OpenAI API configuration.

    Returns:
        True if setup successful, False otherwise
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        st.error("❌ No OpenAI API Key found. Please set OPENAI_API_KEY in .env file")
        return False

    set_api_provider("openai", DEFAULT_OPENAI_MODEL)
    st.success(f"✅ OpenAI API configured with model: **{DEFAULT_OPENAI_MODEL}**")
    return True


def setup_lmstudio_api() -> bool:
    """
    Setup LM Studio API configuration.

    Returns:
        True if setup successful, False otherwise
    """
    # Load available models if not already loaded
    if not has_available_models():
        with st.spinner("Loading available models..."):
            models, error_msg = get_models()
            if not models:
                st.error(f"❌ No models available: {error_msg}")
                return False
            set_available_models(models)

    # Model selection with session state key for persistence
    available_models = get_available_models()
    if not available_models:
        st.error("❌ No models available in LM Studio")
        return False

    selected_model = st.selectbox(
        "Select Model:",
        available_models,
        help="Choose a model loaded in LM Studio",
    )
    set_api_provider("lmstudio", selected_model)
    st.success(f"✅ LM Studio configured with model: **{selected_model}**")
    return True


def render_api_setup_sidebar() -> bool:
    """
    Render API setup in sidebar with provider selection.

    Returns:
        True if setup successful, False otherwise
    """
    st.header("🔧 Setup")

    # Show current model status for debugging
    current_model = get_model_name()
    if current_model:
        st.info(f"**Current Model:** {current_model}")
    else:
        st.warning("⚠️ No model selected")

    api_provider = st.selectbox(
        "Select API Provider:",
        ["OpenAI", "LM Studio"],
        help="Choose between OpenAI API or local LM Studio",
    )

    if api_provider == "OpenAI":
        return setup_openai_api()
    else:  # LM Studio
        return setup_lmstudio_api()
