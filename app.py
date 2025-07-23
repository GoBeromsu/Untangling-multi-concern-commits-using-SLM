import os
import json
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from dotenv import load_dotenv

from utils import llms
from utils.prompt import get_system_prompt
from visual_eval.llms.openai import openai_api_call
from visual_eval.llms.lmstudio import load_model
from visual_eval.llms.constant import (
    CODE_DIFF_INPUT_HEIGHT,
    SYSTEM_PROMPT_INPUT_HEIGHT,
)
from visual_eval.ui.components import (
    render_evaluation_metrics,
    render_results_table,
    create_column_config,
)
from visual_eval.ui.dataset import (
    get_available_datasets,
    load_dataset,
    DIFF_COLUMN,
    TYPES_COLUMN,
    SHAS_COLUMN,
)
from visual_eval.ui.session import (
    get_api_provider,
    get_model_name,
    set_evaluation_results,
)
from visual_eval.ui.setup import render_api_setup_sidebar
from utils.eval import calculate_metrics

# Direct analysis result columns
ANALYSIS_RESULT_COLUMNS = [
    "Predicted_Concern_Types",
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
    "SHAs",
]


def render_system_prompt_input(title: str = "System Prompt") -> str:
    """Render system prompt input widget with consistent styling."""
    st.subheader(title)
    return st.text_area(
        "Modify the system prompt:",
        value=get_system_prompt(),
        height=SYSTEM_PROMPT_INPUT_HEIGHT,
    )


def get_model_response(diff: str, system_prompt: str) -> List[str]:
    """
    Get model response based on selected API provider.

    Args:
        diff: Code diff to analyze
        system_prompt: System prompt for the model

    Returns:
        List of concern types
    """
    selected_api = get_api_provider()

    if selected_api == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model_response = openai_api_call(api_key, diff, system_prompt)
        # Parse OpenAI response (integrated from parse_model_response)
        prediction_data = json.loads(model_response)
        return prediction_data.get("types", [])
    elif selected_api == "lmstudio":
        model_name = get_model_name()
        predicted_types, _ = llms.api_call(model_name, diff, system_prompt)
        return predicted_types
    else:
        raise ValueError(f"Unsupported API provider: {selected_api}")


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


def process_single_case(row: pd.Series, system_prompt: str) -> Dict[str, Any]:
    """Core logic: Process single evaluation case with model prediction."""
    try:
        # Extract data from row
        diff = row[DIFF_COLUMN]
        actual_concern_types = (
            json.loads(row[TYPES_COLUMN]) if row[TYPES_COLUMN] else []
        )
        shas = json.loads(row[SHAS_COLUMN]) if row[SHAS_COLUMN] else []

        # Get model prediction
        predicted_concern_types = get_model_response(diff, system_prompt)

        return {
            "predicted_types": predicted_concern_types,
            "actual_types": actual_concern_types,
            "shas": shas,
            "success": True,
        }
    except Exception as e:
        return {
            "predicted_types": [],
            "actual_types": json.loads(row[TYPES_COLUMN]) if row[TYPES_COLUMN] else [],
            "shas": json.loads(row[SHAS_COLUMN]) if row[SHAS_COLUMN] else [],
            "success": False,
        }


def execute_batch_concern_evaluation(df: pd.DataFrame, system_prompt: str) -> None:
    """Execute batch evaluation using streamlit and pandas delegation."""
    if df.empty:
        st.error("No test data available for evaluation")
        return

    # Pre-load LM Studio model if needed
    selected_api = get_api_provider()
    if selected_api == "lmstudio":
        model_name = get_model_name()
        if model_name:
            with st.spinner(f"Loading model {model_name}..."):
                try:
                    load_model(model_name)
                    st.success(f"✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to load model: {e}")
                    return

    # Process all cases using pandas delegation
    with st.status("Running batch evaluation...", expanded=True) as status:
        st.write(f"Processing {len(df)} test cases...")

        # Use pandas apply for batch processing
        results = []
        progress_bar = st.progress(0)

        for i, (_, row) in enumerate(df.iterrows()):
            case_result = process_single_case(row, system_prompt)
            metrics = calculate_metrics(
                case_result["predicted_types"], case_result["actual_types"]
            )

            # Combine results
            results.append(
                {
                    "Test_Index": i + 1,
                    "Predicted_Types": (
                        ", ".join(sorted(case_result["predicted_types"]))
                        if case_result["predicted_types"]
                        else "None"
                    ),
                    "Actual_Types": (
                        ", ".join(sorted(case_result["actual_types"]))
                        if case_result["actual_types"]
                        else "None"
                    ),
                    "Exact_Match": "Match" if metrics["exact_match"] else "No Match",
                    "Case_Precision": metrics["precision"],
                    "Case_Recall": metrics["recall"],
                    "Case_F1": metrics["f1"],
                    "SHAs": (
                        ", ".join(case_result["shas"])
                        if case_result["shas"]
                        else "None"
                    ),
                }
            )

            progress_bar.progress((i + 1) / len(df))

            # Show warning for failed cases
            if not case_result["success"]:
                st.warning(f"⚠️ API error for case {i+1}")

        # Create results DataFrame using pandas
        evaluation_results_df = pd.DataFrame(results)
        status.update(label="Evaluation complete!", state="complete")

    # Display results
    metrics = calculate_evaluation_metrics(evaluation_results_df)
    render_evaluation_metrics(metrics, len(df))
    render_results_table(evaluation_results_df)

    # Store and download results
    set_evaluation_results(evaluation_results_df)

    if not evaluation_results_df.empty:
        download_df = evaluation_results_df.drop(
            columns=["Case_Precision", "Case_Recall", "Case_F1"]
        )
        csv_data = download_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"evaluation_results_{len(download_df)}_cases.csv",
            mime="text/csv",
            use_container_width=True,
        )


def show_direct_input() -> None:
    """Render UI interface for direct code diff input and concern analysis."""
    with st.form("direct_analysis_form"):
        diff = st.text_area(
            "Enter your code diff:",
            placeholder="Paste the output of `git diff` here...",
            height=CODE_DIFF_INPUT_HEIGHT,
        )

        system_prompt = render_system_prompt_input()

        submitted = st.form_submit_button(
            "Analyze Code Diff", type="primary", use_container_width=True
        )

    if submitted and diff.strip():
        st.divider()
        st.header("📊 Analysis Results")
        with st.spinner("Analyzing code diff..."):
            predicted_concern_types = get_model_response(diff, system_prompt)

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
                }
            )
            st.dataframe(
                analysis_results_df,
                use_container_width=True,
                hide_index=True,
                column_config=create_column_config(ANALYSIS_RESULT_COLUMNS),
            )
    elif submitted and not diff.strip():
        st.warning("Please enter a code diff to analyze.")


def show_csv_input() -> None:
    """Render UI interface for batch evaluation from test dataset files."""
    available_dataset_files = get_available_datasets()

    if not available_dataset_files:
        st.error("No CSV test dataset files found in datasets directory")
        st.stop()

    with st.form("batch_evaluation_form"):
        selected_dataset = st.selectbox(
            "Select concern classification test dataset:",
            available_dataset_files,
            format_func=lambda x: f"{os.path.basename(x)} ({os.path.dirname(x)})",
        )

        system_prompt = render_system_prompt_input("Concern Analysis Prompt Template")

        submitted = st.form_submit_button(
            "Run Batch Evaluation", type="primary", use_container_width=True
        )

    if submitted:
        test_dataset = load_dataset(selected_dataset)
        if not test_dataset.empty:
            st.success(
                f"📊 Loaded **{len(test_dataset)}** test cases from `{selected_dataset}`"
            )
            st.divider()
            st.header("📊 Evaluation Results")
            execute_batch_concern_evaluation(test_dataset, system_prompt)
        else:
            st.error("❌ Failed to load test dataset")


def main() -> None:
    """Main application entry point for concern classification evaluation."""
    st.title("Concern is All You Need")
    load_dotenv()

    # Setup in sidebar
    with st.sidebar:
        setup_success = render_api_setup_sidebar()
        if not setup_success:
            st.stop()

    # Main content
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
