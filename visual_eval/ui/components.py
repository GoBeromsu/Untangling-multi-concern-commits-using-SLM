"""
Reusable UI components for Streamlit interface.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from ..llms.constant import RECENT_RESULTS_DISPLAY_LIMIT


def render_evaluation_metrics(metrics: Dict[str, Any], dataset_size: int) -> None:
    """Render evaluation metrics in a structured layout."""
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        st.metric(
            "Progress",
            f"{metrics['total']}/{dataset_size}",
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


def create_column_config(columns: List[str]) -> Dict[str, Any]:
    """Create standardized column configuration for results table."""
    config_map = {
        "Test_Index": st.column_config.NumberColumn("Test #", width="small"),
        "Predicted_Types": st.column_config.TextColumn(
            "Predicted Types", width="medium"
        ),
        "Actual_Types": st.column_config.TextColumn("Actual Types", width="medium"),
        "Predicted_Count": st.column_config.NumberColumn(
            "Predicted Count", width="small"
        ),
        "Actual_Count": st.column_config.NumberColumn("Actual Count", width="small"),
        "Status": st.column_config.TextColumn("Status", width="small"),
        "Model_Reasoning": st.column_config.TextColumn("Reasoning", width="large"),
        "Predicted_Concern_Types": st.column_config.TextColumn(
            "Predicted Concern Types", width="medium"
        ),
    }
    return {col: config_map[col] for col in columns if col in config_map}


def render_results_table(evaluation_results_df: pd.DataFrame) -> None:
    """Render results table with proper configuration."""
    st.subheader("Evaluation Results")

    display_columns = [
        "Test_Index",
        "Predicted_Types",
        "Actual_Types",
        "Predicted_Count",
        "Actual_Count",
        "Status",
        "Model_Reasoning",
    ]

    # Apply display limit for recent results
    recent_results = (
        evaluation_results_df.tail(RECENT_RESULTS_DISPLAY_LIMIT)
        if len(evaluation_results_df) > RECENT_RESULTS_DISPLAY_LIMIT
        else evaluation_results_df
    )

    display_results_df = recent_results[display_columns]
    column_config = create_column_config(display_columns)

    st.dataframe(
        display_results_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )

    # Add CSV download button for complete results
    if not evaluation_results_df.empty:
        # Prepare full dataset for download (remove internal calculation columns)
        download_df = evaluation_results_df.copy()
        columns_to_exclude = ["Case_Precision", "Case_Recall", "Case_F1"]
        download_columns = [col for col in download_df.columns if col not in columns_to_exclude]
        download_df = download_df[download_columns]
        
        csv_data = download_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv_data,
            file_name=f"concern_evaluation_results_{len(download_df)}_cases.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_dataset_metadata(metadata: Dict[str, Any]) -> None:
    """Render dataset metadata in a structured layout."""
    metadata_col1, metadata_col2, metadata_col3 = st.columns(3)

    with metadata_col1:
        st.caption(f"⚙️ {metadata.get('concerns_per_case', '?')} concerns per case")

    with metadata_col2:
        concern_types_list = ", ".join(metadata.get("types", []))
        st.caption(f"🏷️ {concern_types_list}")

    with metadata_col3:
        types_constraint = (
            "different types"
            if metadata.get("ensure_different_types")
            else "same types allowed"
        )
        st.caption(f"🔀 {types_constraint}")



