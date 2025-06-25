import os
import json
import streamlit as st
from dotenv import load_dotenv
from prompts.type import get_default_prompt_template, get_type_prompt
from llms.openai import openai_api_call


def display_result(result: str) -> None:
    try:
        data = json.loads(result)

        st.subheader("Analysis Results")

        count = data.get("count", 0)
        st.write(f"**Number of concerns:** {count}")

        concern_types = data.get("types", [])
        if concern_types:
            st.write("**Concern types:**")
            for concern_type in concern_types:
                st.write(f"• {concern_type}")
        else:
            st.write("**Concern types:** None identified")

        reason = data.get("reason", "No reasoning provided")
        st.write(f"**Reasoning:** {reason}")

    except json.JSONDecodeError:
        st.error("Failed to parse JSON response")
        st.write("**Raw Response:**")
        st.text(result)
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        st.write("**Raw Response:**")
        st.text(result)


def main() -> None:
    st.title("Concern is All You Need")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        st.success("API Key detected")
    else:
        st.error("No API Key found. Please set OPENAI_API_KEY in .env file")
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
            with st.spinner("Analyzing..."):
                formatted_prompt = get_type_prompt(user_diff, user_prompt)
                result = openai_api_call(api_key, formatted_prompt)
                display_result(result)
        else:
            st.error(
                "Please enter both a diff and a prompt template before submitting."
            )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Concern is All You Need",
        page_icon="🌩️",
        layout="wide",
    )
    main()
