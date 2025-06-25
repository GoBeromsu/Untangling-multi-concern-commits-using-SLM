import os
import json
import glob
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
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


def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["cases"], data.get("metadata", {})
    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        return [], {}


def compare_predictions(predicted_types: List[str], actual_labels: List[str]) -> bool:
    """Compare predicted types with actual labels using exact set comparison."""
    if not predicted_types or not actual_labels:
        return False

    predicted_set = set(predicted_types)
    actual_set = set(actual_labels)

    # Exact set comparison: both sets must contain exactly the same elements
    return predicted_set == actual_set


def run_sequential_test(
    api_key: str, test_data: List[Dict[str, Any]], prompt_template: str
) -> None:
    """Run sequential tests on all test data with real-time table updates."""

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create containers for results
    summary_container = st.empty()
    table_container = st.empty()

    # Initialize results DataFrame
    results_df = pd.DataFrame(
        columns=["Index", "Predicted", "Actual", "P_Count", "A_Count", "Status"]
    )

    for i, item in enumerate(test_data):
        status_text.text(f"Testing item {i+1}/{len(test_data)}...")

        tangle_change = item.get("tangleChange", "")
        atomic_changes = item.get("atomicChanges", [])
        actual_labels = [change.get("label", "") for change in atomic_changes]

        # Get prediction from OpenAI
        formatted_prompt = get_type_prompt(tangle_change, prompt_template)
        response = openai_api_call(api_key, formatted_prompt)

        try:
            prediction_data = json.loads(response)
            predicted_types = prediction_data.get("types", [])
            predicted_count = prediction_data.get("count", 0)
        except json.JSONDecodeError:
            predicted_types = []
            predicted_count = 0

        # Compare with ground truth
        actual_count = len(actual_labels)
        types_match = compare_predictions(predicted_types, actual_labels)
        count_match = predicted_count == actual_count
        is_correct = types_match and count_match

        # Add to DataFrame - intuitive status with types and count indicators
        types_icon = "✅" if types_match else "❌"
        count_icon = "✅" if count_match else "❌"
        status_detail = f"{types_icon}T {count_icon}C"

        new_row = pd.DataFrame(
            {
                "Index": [i + 1],
                "Predicted": [", ".join(predicted_types)],
                "Actual": [", ".join(actual_labels)],
                "P_Count": [predicted_count],
                "A_Count": [actual_count],
                "Status": [status_detail],
            }
        )

        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Update progress
        progress_bar.progress((i + 1) / len(test_data))

        # Update summary
        correct_count = sum(1 for status in results_df["Status"] if "✅" in status)
        with summary_container.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Progress", f"{len(results_df)}/{len(test_data)}")
            with col2:
                st.metric("Correct", correct_count)
            with col3:
                if len(results_df) > 0:
                    accuracy = correct_count / len(results_df)
                    st.metric("Accuracy", f"{accuracy:.1%}")

        with table_container.container():
            st.subheader("Test Results")

            # Show last 15 results for better visibility
            display_df = results_df.tail(15) if len(results_df) > 15 else results_df

            # Configure the dataframe display
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Index": st.column_config.NumberColumn("Test #", width="small"),
                    "Predicted": st.column_config.TextColumn(
                        "Predicted Types", width="medium"
                    ),
                    "Actual": st.column_config.TextColumn(
                        "Actual Types", width="medium"
                    ),
                    "P_Count": st.column_config.NumberColumn("P#", width="small"),
                    "A_Count": st.column_config.NumberColumn("A#", width="small"),
                    "Status": st.column_config.TextColumn("Result", width="medium"),
                },
            )
    st.session_state.final_results_df = results_df


def main() -> None:
    st.title("Concern is All You Need")

    # ========== SETUP SECTION ==========
    st.header("🔧 Setup")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        st.success("✅ OpenAI API Key detected")
    else:
        st.error("❌ No OpenAI API Key found. Please set OPENAI_API_KEY in .env file")
        st.stop()

    st.divider()

    # ========== INPUT SECTION ==========
    st.header("📝 Input")
    tab1, tab2 = st.tabs(["Direct Input", "Test from File"])

    with tab1:
        user_diff = st.text_area(
            "Enter your code diff:",
            placeholder="Paste the output of `git diff` here...",
            height=300,
        )

        st.subheader("Prompt Template")
        default_prompt = get_default_prompt_template()
        prompt_template = st.text_area(
            "Modify the prompt template (use {diff} placeholder):",
            value=default_prompt,
            height=200,
            key="prompt_direct",
        )

        if st.button("Analyze Diff", type="primary", use_container_width=True):
            if user_diff.strip() and prompt_template.strip():
                st.divider()
                st.header("📊 Output")
                with st.spinner("Analyzing..."):
                    formatted_prompt = get_type_prompt(user_diff, prompt_template)
                    result = openai_api_call(api_key, formatted_prompt)
                    display_result(result)
            else:
                st.error("Please enter both a diff and a prompt template.")

    with tab2:
        # Get all JSON files dynamically from datasets directory
        json_files = []
        for pattern in ["datasets/**/*.json"]:
            matches = glob.glob(pattern, recursive=True)
            json_files.extend([f for f in matches if "candidate" not in f])

        # Sort files for consistent display and filter out empty results
        json_files = [f for f in json_files if f.endswith(".json")]
        json_files.sort()

        if not json_files:
            st.error("No JSON files found in datasets directory")
            st.stop()

        test_file = st.selectbox(
            "Select test data file:",
            json_files,
            format_func=lambda x: f"{os.path.basename(x)} ({os.path.dirname(x)})",
        )

        # Auto-load data when file is selected
        if test_file:
            test_data, metadata = load_test_data(test_file)
            if test_data:
                st.success(
                    f"📊 Loaded **{len(test_data)}** test items from `{test_file}`"
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⚙️ {metadata.get('concerns_per_case', '?')} concerns")
                with col2:
                    types = ", ".join(metadata.get("types", []))
                    st.caption(f"🏷️ {types}")
                with col3:
                    st.caption(
                        f"🔀 {'different types' if metadata.get('ensure_different_types') else 'same types allowed'}"
                    )
            else:
                st.error("❌ Failed to load test data")
                test_data = []
        else:
            test_data = []

        st.subheader("Prompt Template")
        default_prompt = get_default_prompt_template()
        prompt_template = st.text_area(
            "Modify the prompt template (use {diff} placeholder):",
            value=default_prompt,
            height=200,
            key="prompt_test",
        )

        if st.button("Run Test", type="primary", use_container_width=True):
            if not test_data:
                st.error("Please select a valid test file first.")
            else:
                st.divider()
                st.header("📊 Output")
                run_sequential_test(api_key, test_data, prompt_template)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Concern is All You Need",
        page_icon="🌩️",
        layout="wide",
    )
    main()
