import os
from typing import Optional
import streamlit as st
from dotenv import load_dotenv


def process_prompt(prompt: str) -> str:
    """Process the input prompt and return output.
    
    Args:
        prompt: User input text.
        
    Returns:
        str: Processed output text.
    """
    if not prompt.strip():
        return "Please enter a valid prompt."
    
    # Simple processing for demonstration
    word_count = len(prompt.split())
    char_count = len(prompt)
    
    return f"""
    **Input Analysis:**
    - Character count: {char_count}
    - Word count: {word_count}
    - First word: {prompt.split()[0] if prompt.split() else 'N/A'}
    
    **Echo Output:**
    {prompt}
    """


if __name__ == "__main__":
    st.set_page_config(
        page_title="Concern is all you need",
        page_icon="💬",
        layout="centered"
    )
    
    load_dotenv()
    api_key_available = os.getenv("OPENAI_API_KEY")
    
    if api_key_available:
        st.success("✅ API Key detected in environment")
    else:
        st.warning("⚠️ No API Key found. Please set OPENAI_API_KEY in .env file")
    
    st.markdown("---")

    with st.form("prompt_form"):
        st.subheader("Input")
        user_prompt = st.text_area(
            "Enter your prompt:",
            placeholder="Type your message here...",
            height=100
        )
        
        submitted = st.form_submit_button("Process")
    
    if submitted:
        st.subheader("Output")
        if user_prompt:
            with st.spinner("Processing..."):
                result = process_prompt(user_prompt)
                st.markdown(result)
        else:
            st.error("Please enter some text before submitting.")