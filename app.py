import os
import json
import glob
import streamlit as st
import pandas as pd
from typing import Dict, Any
from collections import Counter
from dotenv import load_dotenv
from utils.prompt import get_system_prompt
from visual_eval.llms.openai import openai_api_call
from visual_eval.llms.lmstudio import (
    get_models,
    api_call,
    load_model,
)
from visual_eval.llms.constant import (
    CODE_DIFF_INPUT_HEIGHT,
    SYSTEM_PROMPT_INPUT_HEIGHT,
)
from visual_eval.ui.components import (
    render_evaluation_metrics,
    render_results_table,
    create_column_config,
)
from visual_eval.ui.patterns import (
    parse_model_response,
)

# Dataset column constants
DIFF_COLUMN: str = "diff"
TYPES_COLUMN: str = "types"
SHAS_COLUMN: str = "shas"

# File search patterns
CSV_SEARCH_PATTERNS = [
    "datasets/**/*.csv",
    "../datasets/**/*.csv",
]

# Direct analysis result columns
ANALYSIS_RESULT_COLUMNS = [
    "Predicted_Concern_Types",
    "Model_Reasoning",
]

# Evaluation result columns
EVALUATION_RESULT_COLUMNS = [
    "Test_Index",
    "Predicted_Types",
    "Actual_Types",
    "Status",
    "Case_Precision",
    "Case_Recall",
    "Case_F1",
    "Model_Reasoning",
    "SHAs",
]


def get_model_response(diff: str, system_prompt: str) -> str:
    """
    Get model response based on selected API provider.

    Args:
        diff: Code diff to analyze
        system_prompt: System prompt for the model

    Returns:
        JSON string containing the model response
    """
    selected_api = st.session_state.get("selected_api", "openai")

    if selected_api == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return openai_api_call(api_key, diff, system_prompt)
    elif selected_api == "lmstudio":
        model_name = st.session_state.get("selected_model", "")
        return api_call(model_name, diff, system_prompt)
    else:
        raise ValueError(f"Unsupported API provider: {selected_api}")


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load concern classification test dataset from CSV file as DataFrame."""
    try:
        df = pd.read_csv(file_path)

        # Validate required columns exist
        required_columns = [DIFF_COLUMN, TYPES_COLUMN, SHAS_COLUMN]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()

        return df

    except Exception as e:
        st.error(f"Error loading CSV concern test dataset: {str(e)}")
        return pd.DataFrame()


def calculate_evaluation_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics using pre-computed case metrics."""
    total_cases = len(results_df)
    if total_cases == 0:
        return {
            "total": 0,
            "type_precision": 0.0,
            "type_recall": 0.0,
            "type_f1_macro": 0.0,
        }

    # Macro-average: simply average the pre-calculated case metrics
    macro_precision = results_df["Case_Precision"].mean()
    macro_recall = results_df["Case_Recall"].mean()
    macro_f1 = results_df["Case_F1"].mean()

    return {
        "total": total_cases,
        "type_precision": macro_precision,
        "type_recall": macro_recall,
        "type_f1_macro": macro_f1,
    }


def execute_batch_concern_evaluation(df: pd.DataFrame, system_prompt: str) -> None:
    """Execute batch evaluation of concern classification with real-time updates using DataFrame."""

    if df.empty:
        st.error("No test data available for evaluation")
        return

    # Pre-load model if using LM Studio to prevent loading during evaluation
    selected_api = st.session_state.get("selected_api", "openai")
    if selected_api == "lmstudio":
        model_name = st.session_state.get("selected_model", "")
        if model_name:
            with st.spinner(f"Loading model {model_name} before evaluation..."):
                try:
                    load_model(model_name)
                    st.success(f"✅ Model {model_name} loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to load model: {e}")
                    return

    progress_bar = st.progress(0)
    status_text = st.empty()
    summary_container = st.empty()
    table_container = st.empty()

    # Initialize evaluation results DataFrame using constants (pre-allocate rows)
    evaluation_results_df = pd.DataFrame(
        index=df.index, columns=EVALUATION_RESULT_COLUMNS
    )

    total_cases = len(df)

    for test_index, row in df.iterrows():
        status_text.text(f"Evaluating case {test_index+1}/{total_cases}...")

        # Extract data directly from DataFrame row using constants
        diff = row[DIFF_COLUMN]

        # Parse JSON strings to get actual lists
        actual_concern_types = (
            json.loads(row[TYPES_COLUMN]) if row[TYPES_COLUMN] else []
        )
        shas = json.loads(row[SHAS_COLUMN]) if row[SHAS_COLUMN] else []

        # Get model prediction with error handling
        try:
            model_response = get_model_response(diff, system_prompt)
            predicted_concern_types, model_reasoning = parse_model_response(model_response)
        except Exception as e:
            # Continue with next case when API response fails
            predicted_concern_types = []
            model_reasoning = f"API Error: {str(e)}"
            st.warning(f"⚠️ API error for case {test_index+1}: {str(e)}")

        # Calculate evaluation results using Counter once (linear approach)
        predicted_counter = Counter(predicted_concern_types)
        actual_counter = Counter(actual_concern_types)

        # Type matching
        concern_types_match = predicted_counter == actual_counter

        # STRICT MULTISET MATCHING: Calculate TP, FP, FN with exact count requirements
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
        evaluation_status = f"{types_status_icon}T"

        # Format SHA information as comma-separated string
        formatted_shas = ", ".join(shas) if shas else "None"

        # Update the pre-allocated row for current test_index
        evaluation_results_df.loc[
            test_index,
            [
                "Test_Index",
                "Predicted_Types",
                "Actual_Types",
                "Status",
                "Model_Reasoning",
                "Case_Precision",
                "Case_Recall",
                "Case_F1",
                "SHAs",
            ],
        ] = [
            test_index + 1,
            (
                ", ".join(sorted(predicted_concern_types))
                if predicted_concern_types
                else "None"
            ),
            ", ".join(sorted(actual_concern_types)) if actual_concern_types else "None",
            evaluation_status,
            model_reasoning,
            case_precision,
            case_recall,
            case_f1,
            formatted_shas,
        ]
        progress_bar.progress((test_index + 1) / total_cases)

        # Update evaluation summary (consider only processed rows)
        processed_df = evaluation_results_df.dropna(subset=["Status"], how="any")
        metrics = calculate_evaluation_metrics(processed_df)
        with summary_container.container():
            render_evaluation_metrics(metrics, total_cases)

        # Update results table (exclude case metrics from display)
        with table_container.container():
            render_results_table(evaluation_results_df)

    # Store final results for analysis
    st.session_state.final_evaluation_results = evaluation_results_df

    # Single download button for full results CSV
    if not evaluation_results_df.empty:
        download_df = evaluation_results_df.copy()
        columns_to_exclude = ["Case_Precision", "Case_Recall", "Case_F1"]
        download_columns = [
            col for col in download_df.columns if col not in columns_to_exclude
        ]
        download_df = download_df[download_columns]
        csv_data = download_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv_data,
            file_name=f"concern_evaluation_results_{len(download_df)}_cases.csv",
            mime="text/csv",
            key="download_full_results",
            use_container_width=True,
        )


def show_direct_input() -> None:
    """Render UI interface for direct code diff input and concern analysis."""
    diff = st.text_area(
        "Enter your code diff:",
        placeholder="Paste the output of `git diff` here...",
        height=CODE_DIFF_INPUT_HEIGHT,
    )

    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "Modify the system prompt:",
        value=get_system_prompt(),
        height=SYSTEM_PROMPT_INPUT_HEIGHT,
        key="direct_input_prompt",
    )

    if st.button("Analyze Code Diff", type="primary", use_container_width=True):
        st.divider()
        st.header("📊 Analysis Results")
        with st.spinner("Analyzing code diff..."):
            model_response = get_model_response(diff, system_prompt)

            predicted_concern_types, model_reasoning = parse_model_response(
                model_response
            )

            if model_reasoning == "Failed to parse model response":
                st.error("Failed to parse model response")
                st.write("**Raw Model Response:**")
                st.text(model_response)
            else:
                st.subheader("Concern Classification Results")

                analysis_results_df = pd.DataFrame(
                    {
                        "Predicted_Concern_Types": [
                            (
                                ", ".join(sorted(predicted_concern_types))
                                if predicted_concern_types
                                else "None"
                            )
                        ],
                        "Model_Reasoning": [model_reasoning],
                    }
                )
                column_config = create_column_config(ANALYSIS_RESULT_COLUMNS)

                st.dataframe(
                    analysis_results_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config,
                )


def show_csv_input() -> None:
    """Render UI interface for batch evaluation from test dataset files."""
    available_dataset_files = []
    for search_pattern in CSV_SEARCH_PATTERNS:
        matched_files = glob.glob(search_pattern, recursive=True)
        available_dataset_files.extend([f for f in matched_files])
    available_dataset_files.sort()

    if not available_dataset_files:
        st.error("No CSV test dataset files found in datasets directory")
        st.stop()

    selected_dataset = st.selectbox(
        "Select concern classification test dataset:",
        available_dataset_files,
        format_func=lambda x: f"{os.path.basename(x)} ({os.path.dirname(x)})",
    )

    test_dataset = pd.DataFrame()
    if selected_dataset:
        test_dataset = load_dataset(selected_dataset)
        if not test_dataset.empty:
            st.success(
                f"📊 Loaded **{len(test_dataset)}** test cases from `{selected_dataset}`"
            )
        else:
            st.error("❌ Failed to load test dataset")

    st.subheader("Concern Analysis Prompt Template")
    system_prompt = st.text_area(
        "Modify the prompt template (use {diff} placeholder):",
        value=get_system_prompt(),
        height=SYSTEM_PROMPT_INPUT_HEIGHT,
        key="batch_evaluation_prompt",
    )

    if st.button("Run Batch Evaluation", type="primary", use_container_width=True):
        if test_dataset.empty:
            st.error("Please select a valid test dataset file first.")
        else:
            st.divider()
            st.header("📊 Evaluation Results")
            execute_batch_concern_evaluation(test_dataset, system_prompt)


def main() -> None:
    """Main application entry point for concern classification evaluation."""
    st.title("Concern is All You Need")

    st.header("🔧 Setup")
    load_dotenv()

    # API Provider Selection
    api_provider = st.selectbox(
        "Select API Provider:",
        ["OpenAI", "LM Studio"],
        help="Choose between OpenAI API or local LM Studio",
    )

    # Handle OpenAI API setup
    if api_provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            st.error(
                "❌ No OpenAI API Key found. Please set OPENAI_API_KEY in .env file"
            )
            st.stop()

        st.success("✅ OpenAI API Key detected")
        st.session_state.update({"selected_api": "openai", "selected_model": None})

    # Handle LM Studio setup
    else:  # api_provider == "LM Studio"
        if "lmstudio_available_models" not in st.session_state:
            with st.spinner("Loading available models..."):
                models, error_msg = get_models()
                if not models:
                    st.error(f"❌ No models available: {error_msg}")
                    st.stop()
                st.session_state.lmstudio_available_models = models

        selected_model = st.selectbox(
            "Select Model:",
            st.session_state.lmstudio_available_models,
            help="Choose a model loaded in LM Studio",
        )

        st.session_state.update(
            {
                "selected_api": "lmstudio",
                "selected_model": selected_model,
            }
        )

    st.divider()

    st.header("📝 Concern Classification Analysis")
    direct_input_tab, batch_evaluation_tab = st.tabs(
        ["Direct Code Diff Analysis", "Batch Dataset Evaluation"]
    )

    with direct_input_tab:
        show_direct_input()
    with batch_evaluation_tab:
        show_csv_input()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Concern is All You Need",
        page_icon="🌩️",
        layout="wide",
    )
    main()
