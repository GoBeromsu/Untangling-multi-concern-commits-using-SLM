import os
import json
import glob
import streamlit as st
import pandas as pd
from typing import Dict, Any

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
from utils.eval import calculate_metrics

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


def render_system_prompt_input(title: str = "System Prompt") -> str:
    """Render system prompt input widget with consistent styling."""
    st.subheader(title)
    return st.text_area(
        "Modify the system prompt:",
        value=get_system_prompt(),
        height=SYSTEM_PROMPT_INPUT_HEIGHT,
    )


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


@st.cache_data
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
        model_response = get_model_response(diff, system_prompt)
        predicted_concern_types, model_reasoning = parse_model_response(model_response)

        return {
            "predicted_types": predicted_concern_types,
            "actual_types": actual_concern_types,
            "model_reasoning": model_reasoning,
            "shas": shas,
            "success": True,
        }
    except Exception as e:
        return {
            "predicted_types": [],
            "actual_types": json.loads(row[TYPES_COLUMN]) if row[TYPES_COLUMN] else [],
            "model_reasoning": f"API Error: {str(e)}",
            "shas": json.loads(row[SHAS_COLUMN]) if row[SHAS_COLUMN] else [],
            "success": False,
        }


def execute_batch_concern_evaluation(df: pd.DataFrame, system_prompt: str) -> None:
    """Execute batch evaluation using streamlit and pandas delegation."""
    if df.empty:
        st.error("No test data available for evaluation")
        return

    # Pre-load LM Studio model if needed
    selected_api = st.session_state.get("selected_api", "openai")
    if selected_api == "lmstudio":
        model_name = st.session_state.get("selected_model", "")
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
                    "Status": "✅T" if metrics["exact_match"] else "❌T",
                    "Model_Reasoning": case_result["model_reasoning"],
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
    st.session_state.final_evaluation_results = evaluation_results_df

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
    available_dataset_files = []
    for search_pattern in CSV_SEARCH_PATTERNS:
        matched_files = glob.glob(search_pattern, recursive=True)
        available_dataset_files.extend([f for f in matched_files])
    available_dataset_files.sort()

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
        st.header("🔧 Setup")
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
                {"selected_api": "lmstudio", "selected_model": selected_model}
            )

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
