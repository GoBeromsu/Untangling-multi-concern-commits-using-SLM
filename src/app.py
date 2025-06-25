import os
from typing import Optional
import streamlit as st
from dotenv import load_dotenv

from llms.openai import generate_completion
from prompts.type import get_default_prompt_template, get_type_prompt


def main() -> None:
    st.title("Concern is all you need")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        st.success("✅ API Key detected in environment")
    else:
        st.error("⚠️ No API Key found. Please set OPENAI_API_KEY in .env file")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Diff Input")
        user_diff = st.text_area(
            "Enter your code diff here:",
            placeholder="Paste the output of `git diff` here...",
            height=350,
            key="diff_input",
        )

    with col2:
        st.subheader("Prompt Template")
        default_prompt = get_default_prompt_template()
        user_prompt = st.text_area(
            "Modify the prompt template (use {diff} placeholder):",
            value=default_prompt,
            height=350,
            key="prompt_input",
        )

    if st.button("Analyze Diff", type="primary"):
        if user_diff.strip() and user_prompt.strip():
            with st.spinner("Analyzing diff..."):
                formatted_prompt = get_type_prompt(user_diff, user_prompt)

                result = generate_completion(api_key, formatted_prompt)

                st.subheader("Analysis Result")
                st.markdown(result)
        else:
            st.error(
                "Please enter both a diff and a prompt template before submitting."
            )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Concern is all you need",
        page_icon="🌩️",
        layout="wide",
    )
    main()
