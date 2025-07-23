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
    # Try to load available models (simple approach like main.py)
    if not has_available_models():
        with st.spinner("Loading available models..."):
            try:
                models, error_msg = get_models()
                
                if models:  # Success - we have models
                    set_available_models(models)
                    st.success(f"✅ Found {len(models)} available models")
                else:
                    # Show error but also provide manual option
                    if error_msg:
                        st.warning(f"⚠️ Auto-detection failed: {error_msg}")
                    st.info("💡 You can manually enter a model name below")
                    
                    # Manual model input as fallback
                    manual_model = st.text_input(
                        "Manual Model Name:",
                        placeholder="e.g., microsoft/phi-4",
                        help="Enter the exact model name as it appears in LM Studio"
                    )
                    
                    if manual_model.strip():
                        set_available_models([manual_model.strip()])
                        st.info(f"Using manual model: {manual_model.strip()}")
                    else:
                        return False
                        
            except Exception as e:
                st.error(f"❌ LM Studio setup failed: {str(e)}")
                st.info("💡 Make sure LM Studio is running with a model loaded")
                return False

    # Model selection
    available_models = get_available_models()
    if not available_models:
        return False

    if len(available_models) == 1:
        selected_model = available_models[0]
        st.info(f"Using model: **{selected_model}**")
    else:
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
