import os
import json
import glob
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import Counter
from dotenv import load_dotenv
from prompts.type import get_system_prompt
from llms.openai import openai_api_call


def load_concern_test_dataset(
    file_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load concern classification test dataset from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["cases"], data.get("metadata", {})
    except Exception as e:
        st.error(f"Error loading concern test dataset: {str(e)}")
        return [], {}


def format_concern_types(concern_types: List[str]) -> str:
    """Format concern types list for table display, preserving duplicates."""
    if not concern_types:
        return "None"

    # Sort for consistent display but preserve duplicates
    return ", ".join(sorted(concern_types))


def calculate_evaluation_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics using pre-computed case metrics."""
    total_cases = len(results_df)
    if total_cases == 0:
        return {
            "total": 0,
            "type_precision": 0.0,
            "type_recall": 0.0,
            "type_f1_macro": 0.0,
            "count_accuracy": 0.0,
        }

    # Count prediction accuracy: exact match between predicted and actual counts
    count_correct = sum(
        1
        for _, row in results_df.iterrows()
        if row["Predicted_Count"] == row["Actual_Count"]
    )
    count_accuracy = count_correct / total_cases

    # Macro-average: simply average the pre-calculated case metrics
    macro_precision = results_df["Case_Precision"].mean()
    macro_recall = results_df["Case_Recall"].mean()
    macro_f1 = results_df["Case_F1"].mean()

    return {
        "total": total_cases,
        "type_precision": macro_precision,
        "type_recall": macro_recall,
        "type_f1_macro": macro_f1,
        "count_accuracy": count_accuracy,
    }


def execute_batch_concern_evaluation(
    api_key: str, test_dataset: List[Dict[str, Any]], system_prompt: str
) -> None:
    """Execute batch evaluation of concern classification with real-time updates."""

    progress_bar = st.progress(0)
    status_text = st.empty()
    summary_container = st.empty()
    table_container = st.empty()

    # Initialize evaluation results DataFrame
    evaluation_results_df = pd.DataFrame(
        columns=[
            "Test_Index",
            "Predicted_Types",
            "Actual_Types",
            "Predicted_Count",
            "Actual_Count",
            "Status",
            "Case_Precision",
            "Case_Recall",
            "Case_F1",
            "Model_Reasoning",
        ]
    )

    for test_index, test_case in enumerate(test_dataset):
        status_text.text(f"Evaluating case {test_index+1}/{len(test_dataset)}...")

        diff = test_case.get("tangleChange", "")
        atomic_changes = test_case.get("atomicChanges", [])
        actual_concern_types = [change.get("label", "") for change in atomic_changes]

        # Get model prediction
        model_response = openai_api_call(api_key, diff, system_prompt)

        try:
            prediction_data = json.loads(model_response)
            predicted_concern_types = prediction_data.get("types", [])
            predicted_concern_count = prediction_data.get("count", 0)
            model_reasoning = prediction_data.get("reason", "No reasoning provided")
        except json.JSONDecodeError:
            predicted_concern_types = []
            predicted_concern_count = 0
            model_reasoning = "Failed to parse model response"

        # Calculate evaluation results using Counter once (linear approach)
        actual_concern_count = len(actual_concern_types)

        # Single Counter calculation for all metrics
        predicted_counter = Counter(predicted_concern_types)
        actual_counter = Counter(actual_concern_types)

        # Type matching and count matching
        concern_types_match = predicted_counter == actual_counter
        concern_count_match = predicted_concern_count == actual_concern_count

        # STRICT MULTISET MATCHING: Calculate TP, FP, FN with exact count requirements
        # Example: if actual=['feat'] but predicted=['feat','feat'], then TP=1, FP=1, FN=0
        # This means: feat→feat,feat is WRONG (over-prediction causes FP)
        all_types = set(predicted_counter.keys()) | set(actual_counter.keys())
        tp = sum(
            min(predicted_counter[t], actual_counter[t]) for t in all_types
        )  # Correctly matched count
        fp = sum(
            max(0, predicted_counter[t] - actual_counter[t]) for t in all_types
        )  # Over-predicted count
        fn = sum(
            max(0, actual_counter[t] - predicted_counter[t]) for t in all_types
        )  # Under-predicted count

        # Calculate individual case metrics
        case_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        case_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        case_f1 = (
            2 * (case_precision * case_recall) / (case_precision + case_recall)
            if (case_precision + case_recall) > 0
            else 0.0
        )

        # Create evaluation status display
        types_status_icon = "✅" if concern_types_match else "❌"
        count_status_icon = "✅" if concern_count_match else "❌"
        evaluation_status = f"{types_status_icon}T {count_status_icon}C"

        # Add result to DataFrame with case metrics for internal calculation
        new_evaluation_result = pd.DataFrame(
            {
                "Test_Index": [test_index + 1],
                "Predicted_Types": [format_concern_types(predicted_concern_types)],
                "Actual_Types": [format_concern_types(actual_concern_types)],
                "Predicted_Count": [predicted_concern_count],
                "Actual_Count": [actual_concern_count],
                "Status": [evaluation_status],
                "Model_Reasoning": [model_reasoning],
                "Case_Precision": [case_precision],
                "Case_Recall": [case_recall],
                "Case_F1": [case_f1],
            }
        )

        evaluation_results_df = pd.concat(
            [evaluation_results_df, new_evaluation_result], ignore_index=True
        )
        progress_bar.progress((test_index + 1) / len(test_dataset))

        # Update evaluation summary
        metrics = calculate_evaluation_metrics(evaluation_results_df)
        with summary_container.container():
            # Evaluation metrics display: Progress and Type Classification metrics
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
            with col1:
                st.metric(
                    "Progress",
                    f"{metrics['total']}/{len(test_dataset)}",
                    help="Current evaluation progress through test dataset",
                )
            with col2:
                st.metric(
                    "Type Precision",
                    f"{metrics['type_precision']:.1%}",
                    help="Sample-based precision using strict multiset matching: average of per-case precision scores. Over-prediction (feat→feat,feat) causes FP penalty.",
                )
            with col3:
                st.metric(
                    "Type Recall",
                    f"{metrics['type_recall']:.1%}",
                    help="Sample-based recall using strict multiset matching: average of per-case recall scores. Under-prediction causes FN penalty.",
                )
            with col4:
                st.metric(
                    "Type F1 (Macro)",
                    f"{metrics['type_f1_macro']:.1%}",
                    help="Sample-based F1 using strict multiset matching: average of per-case F1 scores. Exact count matching required (feat ≠ feat,feat).",
                )
            with col5:
                st.metric(
                    "Count Accuracy",
                    f"{metrics['count_accuracy']:.1%}",
                    help="Percentage of test cases with exact count match between predicted and actual",
                )

        # Update results table (exclude case metrics from display)
        with table_container.container():
            st.subheader("Evaluation Results")
            # Select only display columns (exclude case metrics)
            display_columns = [
                "Test_Index",
                "Predicted_Types",
                "Actual_Types",
                "Predicted_Count",
                "Actual_Count",
                "Status",
                "Model_Reasoning",
            ]
            recent_results = (
                evaluation_results_df.tail(15)
                if len(evaluation_results_df) > 15
                else evaluation_results_df
            )
            display_results_df = recent_results[display_columns]

            column_config = {
                "Test_Index": st.column_config.NumberColumn("Test #", width="small"),
                "Predicted_Types": st.column_config.TextColumn(
                    "Predicted Types", width="medium"
                ),
                "Actual_Types": st.column_config.TextColumn(
                    "Actual Types", width="medium"
                ),
                "Predicted_Count": st.column_config.NumberColumn(
                    "Predicted Count", width="small"
                ),
                "Actual_Count": st.column_config.NumberColumn(
                    "Actual Count", width="small"
                ),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Model_Reasoning": st.column_config.TextColumn(
                    "Reasoning", width="large"
                ),
            }

            st.dataframe(
                display_results_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
            )

    # Store final results for analysis
    st.session_state.final_evaluation_results = evaluation_results_df


def render_direct_input_interface(api_key: str) -> None:
    """Render UI interface for direct code diff input and concern analysis."""
    diff = st.text_area(
        "Enter your code diff:",
        placeholder="Paste the output of `git diff` here...",
        height=300,
    )

    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "Modify the system prompt:",
        value=get_system_prompt(),
        height=200,
        key="direct_input_prompt",
    )

    if st.button("Analyze Code Diff", type="primary", use_container_width=True):
        st.divider()
        st.header("📊 Analysis Results")
        with st.spinner("Analyzing code diff..."):
            model_response = openai_api_call(api_key, diff, system_prompt)

            try:
                analysis_data = json.loads(model_response)
                predicted_concern_types = analysis_data.get("types", [])
                predicted_concern_count = analysis_data.get("count", 0)
                model_reasoning = analysis_data.get("reason", "No reasoning provided")

                st.subheader("Concern Classification Results")

                analysis_results_df = pd.DataFrame(
                    {
                        "Predicted_Concern_Types": [
                            format_concern_types(predicted_concern_types)
                        ],
                        "Predicted_Count": [predicted_concern_count],
                        "Model_Reasoning": [model_reasoning],
                    }
                )

                column_config = {
                    "Predicted_Concern_Types": st.column_config.TextColumn(
                        "Predicted Concern Types", width="medium"
                    ),
                    "Predicted_Count": st.column_config.NumberColumn(
                        "Count", width="small"
                    ),
                    "Model_Reasoning": st.column_config.TextColumn(
                        "Reasoning", width="large"
                    ),
                }

                st.dataframe(
                    analysis_results_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config,
                )

            except json.JSONDecodeError:
                st.error("Failed to parse model response")
                st.write("**Raw Model Response:**")
                st.text(model_response)
            except Exception as e:
                st.error(f"Error processing analysis results: {str(e)}")
                st.write("**Raw Model Response:**")
                st.text(model_response)


def render_batch_evaluation_interface(api_key: str) -> None:
    """Render UI interface for batch evaluation from test dataset files."""
    available_dataset_files = []
    for search_pattern in ["datasets/**/*.json"]:
        matched_files = glob.glob(search_pattern, recursive=True)
        available_dataset_files.extend(
            [f for f in matched_files if "candidate" not in f]
        )

    available_dataset_files = [
        f for f in available_dataset_files if f.endswith(".json")
    ]
    available_dataset_files.sort()

    if not available_dataset_files:
        st.error("No test dataset files found in datasets directory")
        st.stop()

    selected_dataset_file = st.selectbox(
        "Select concern classification test dataset:",
        available_dataset_files,
        format_func=lambda x: f"{os.path.basename(x)} ({os.path.dirname(x)})",
    )

    if selected_dataset_file:
        test_dataset, dataset_metadata = load_concern_test_dataset(
            selected_dataset_file
        )
        if test_dataset:
            st.success(
                f"📊 Loaded **{len(test_dataset)}** test cases from `{selected_dataset_file}`"
            )
            metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
            with metadata_col1:
                st.caption(
                    f"⚙️ {dataset_metadata.get('concerns_per_case', '?')} concerns per case"
                )
            with metadata_col2:
                concern_types_list = ", ".join(dataset_metadata.get("types", []))
                st.caption(f"🏷️ {concern_types_list}")
            with metadata_col3:
                types_constraint = (
                    "different types"
                    if dataset_metadata.get("ensure_different_types")
                    else "same types allowed"
                )
                st.caption(f"🔀 {types_constraint}")
        else:
            st.error("❌ Failed to load test dataset")
            test_dataset = []
    else:
        test_dataset = []

    st.subheader("Concern Analysis Prompt Template")
    system_prompt = st.text_area(
        "Modify the prompt template (use {diff} placeholder):",
        value=get_system_prompt(),
        height=200,
        key="batch_evaluation_prompt",
    )

    if st.button("Run Batch Evaluation", type="primary", use_container_width=True):
        if not test_dataset:
            st.error("Please select a valid test dataset file first.")
        else:
            st.divider()
            st.header("📊 Evaluation Results")
            execute_batch_concern_evaluation(api_key, test_dataset, system_prompt)


def main() -> None:
    """Main application entry point for concern classification evaluation."""
    st.title("Concern is All You Need")

    st.header("🔧 Setup")
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key:
        st.success("✅ OpenAI API Key detected")
    else:
        st.error("❌ No OpenAI API Key found. Please set OPENAI_API_KEY in .env file")
        st.stop()

    st.divider()

    st.header("📝 Concern Classification Analysis")
    direct_input_tab, batch_evaluation_tab = st.tabs(
        ["Direct Code Diff Analysis", "Batch Dataset Evaluation"]
    )

    with direct_input_tab:
        render_direct_input_interface(openai_api_key)
    with batch_evaluation_tab:
        render_batch_evaluation_interface(openai_api_key)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Concern is All You Need",
        page_icon="🌩️",
        layout="wide",
    )
    main()
