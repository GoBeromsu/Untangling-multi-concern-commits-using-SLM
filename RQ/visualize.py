#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

import utils.eval as eval_utils

RESULTS_BASE_DIR = Path("results")
ANALYSIS_OUTPUT_DIR = Path("results/analysis")
CONTEXT_LENGTH = [1024, 2048, 4096, 8192, 12288]
EXPERIMENT_TYPES = ["with_message", "diff_only"]
METRICS = ["accuracy", "f1", "precision", "recall"]

plt.style.use("default")
sns.set_palette("husl")


def parse_filename(filepath: Path) -> Tuple[str, int]:
    """
    Parse model name and context length from filename.
    Example: microsoft_phi-4_4096.csv -> ('microsoft_phi-4', 4096)
    """
    filename = filepath.stem  # Remove .csv extension
    parts = filename.split("_")

    # Get context_len from the last part
    context_len = int(parts[-1])

    # Get model name by joining all parts except the last one
    model = "_".join(parts[:-1])

    return model, context_len


def preprocess_experimental_data() -> pd.DataFrame:
    """
    Load experimental data and add metadata columns.
    Returns DataFrame with: predicted_types, actual_types, inference_time, shas, context_len, with_message, model
    """
    all_dataframes = []

    for experiment_type in EXPERIMENT_TYPES:
        exp_folder = RESULTS_BASE_DIR / experiment_type

        if not exp_folder.exists():
            continue

        with_message = 1 if experiment_type == "with_message" else 0

        for csv_file in exp_folder.glob("*.csv"):
            try:
                model, context_len = parse_filename(csv_file)
            except (ValueError, IndexError):
                continue

            df = pd.read_csv(csv_file)

            # Add only the required metadata columns
            df["context_len"] = context_len
            df["with_message"] = with_message
            df["model"] = model

            all_dataframes.append(df)
    result = pd.concat(all_dataframes, ignore_index=True)
    return result


def create_metrics_by_context_window(df: pd.DataFrame) -> None:
    """Create visualization of metrics by context window using calculated metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(METRICS):
        ax = axes[i]

        # Plot for each experiment type
        for exp_type in EXPERIMENT_TYPES:
            data = df[df["experiment_type"] == exp_type]
            if data.empty:
                continue

            # Group by context window and calculate mean
            grouped = data.groupby("context_window")[metric].mean().reset_index()

            ax.plot(
                grouped["context_window"],
                grouped[metric],
                marker="o",
                label=exp_type.replace("_", " ").title(),
                linewidth=2,
                markersize=8,
            )

        ax.set_title(f"{metric.title()} by Context Window", fontsize=14)
        ax.set_xlabel("Context Window Size", fontsize=12)
        ax.set_ylabel(metric.title(), fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

    plt.tight_layout()
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        ANALYSIS_OUTPUT_DIR / "metrics_by_context_window.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save as table using eval_utils approach
    table_data = []
    for exp_type in EXPERIMENT_TYPES:
        for context_window in CONTEXT_LENGTH:
            data = df[
                (df["experiment_type"] == exp_type)
                & (df["context_window"] == context_window)
            ]
            if not data.empty:
                row = {"experiment_type": exp_type, "context_window": context_window}
                for metric in METRICS:
                    row[metric] = data[metric].mean()
                table_data.append(row)

    table_df = pd.DataFrame(table_data)
    eval_utils.save_results(
        table_df, {}, str(ANALYSIS_OUTPUT_DIR / "metrics_by_context_window")
    )


def create_metrics_by_concern_count(df: pd.DataFrame) -> None:
    """Create visualization of metrics by concern count using actual_types length."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(METRICS):
        ax = axes[i]

        # Plot for each experiment type
        for exp_type in EXPERIMENT_TYPES:
            data = df[df["experiment_type"] == exp_type]
            if data.empty:
                continue

            # Group by actual concern count (calculated from actual_types)
            grouped = data.groupby("actual_count")[metric].mean().reset_index()

            ax.plot(
                grouped["actual_count"],
                grouped[metric],
                marker="o",
                label=exp_type.replace("_", " ").title(),
                linewidth=2,
                markersize=8,
            )

        ax.set_title(f"{metric.title()} by Actual Concern Count", fontsize=14)
        ax.set_xlabel("Number of Concerns", fontsize=12)
        ax.set_ylabel(metric.title(), fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3])

    plt.tight_layout()
    plt.savefig(
        ANALYSIS_OUTPUT_DIR / "metrics_by_concern_count.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save as table using eval_utils approach
    table_data = []
    for exp_type in EXPERIMENT_TYPES:
        for concern_count in [1, 2, 3]:
            data = df[
                (df["experiment_type"] == exp_type)
                & (df["actual_count"] == concern_count)
            ]
            if not data.empty:
                row = {"experiment_type": exp_type, "concern_count": concern_count}
                for metric in METRICS:
                    row[metric] = data[metric].mean()
                table_data.append(row)

    table_df = pd.DataFrame(table_data)
    eval_utils.save_results(
        table_df, {}, str(ANALYSIS_OUTPUT_DIR / "metrics_by_concern_count")
    )


def create_comparison_plots(df: pd.DataFrame) -> None:
    """Create comparison plots between with_message and diff_only using calculated metrics."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Calculate average metrics for each experiment type using utils approach
    comparison_data = []
    for exp_type in EXPERIMENT_TYPES:
        data = df[df["experiment_type"] == exp_type]
        if not data.empty:
            # Use calculate_batch_metrics for overall performance
            batch_metrics = eval_utils.calculate_batch_metrics(
                data[["predicted_types_parsed", "actual_types_parsed"]].rename(
                    columns={
                        "predicted_types_parsed": "predictions",
                        "actual_types_parsed": "ground_truth",
                    }
                )
            )
            for metric in ["f1", "precision", "recall"]:
                comparison_data.append(
                    {
                        "experiment_type": exp_type,
                        "metric": metric,
                        "value": batch_metrics[metric],
                    }
                )
            # Add accuracy manually
            comparison_data.append(
                {
                    "experiment_type": exp_type,
                    "metric": "accuracy",
                    "value": data["accuracy"].mean(),
                }
            )

    comparison_df = pd.DataFrame(comparison_data)

    # Create grouped bar plot
    pivot = comparison_df.pivot(
        index="metric", columns="experiment_type", values="value"
    )
    pivot.plot(kind="bar", ax=ax, width=0.8)

    ax.set_title("Average Metrics Comparison: With Message vs Diff Only", fontsize=16)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Average Score", fontsize=12)
    ax.legend(title="Experiment Type", labels=["Diff Only", "With Message"])
    ax.grid(True, alpha=0.3, axis="y")

    # Rotate x-labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(
        ANALYSIS_OUTPUT_DIR / "metrics_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save comparison table using eval_utils
    eval_utils.save_results(
        pivot.reset_index(), {}, str(ANALYSIS_OUTPUT_DIR / "metrics_comparison")
    )


def create_inference_time_vs_metrics(df: pd.DataFrame) -> None:
    """Create linear visualization showing inference time vs all metrics (concern count agnostic)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Group by inference time bins and calculate mean for each metric
    df_sorted = df.sort_values("inference_time")

    # Create 15 equal-sized bins for smoother trend lines
    num_bins = 15
    bin_size = len(df_sorted) // num_bins

    for metric in METRICS:
        time_points = []
        metric_points = []

        for i in range(0, len(df_sorted), bin_size):
            bin_data = df_sorted.iloc[i : i + bin_size]
            if len(bin_data) > 0:
                time_points.append(bin_data["inference_time"].mean())
                metric_points.append(bin_data[metric].mean())

        ax.plot(
            time_points,
            metric_points,
            marker="o",
            label=metric.title(),
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Inference Time (seconds)", fontsize=12)
    ax.set_ylabel("Metric Score", fontsize=12)
    ax.set_title("Inference Time vs All Metrics (All Concerns)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        ANALYSIS_OUTPUT_DIR / "inference_time_vs_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_context_window_vs_metrics(df: pd.DataFrame) -> None:
    """Create linear visualization showing context window vs all metrics (concern count agnostic)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Group by context window and calculate mean for each metric
    for metric in METRICS:
        context_metrics = df.groupby("context_window")[metric].mean().reset_index()

        ax.plot(
            context_metrics["context_window"],
            context_metrics[metric],
            marker="o",
            linewidth=2,
            markersize=8,
            label=metric.title(),
        )

    ax.set_xlabel("Context Window Size", fontsize=12)
    ax.set_ylabel("Metric Score", fontsize=12)
    ax.set_title("Context Window vs All Metrics (All Concerns)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()
    plt.savefig(
        ANALYSIS_OUTPUT_DIR / "context_window_vs_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save as table with all metrics
    table_data = []
    for context_window in CONTEXT_LENGTH:
        row = {"context_window": context_window}
        for metric in METRICS:
            context_data = df[df["context_window"] == context_window]
            if not context_data.empty:
                row[metric] = context_data[metric].mean()
        table_data.append(row)

    context_metrics_table = pd.DataFrame(table_data)
    eval_utils.save_results(
        context_metrics_table,
        {},
        str(ANALYSIS_OUTPUT_DIR / "context_window_vs_metrics"),
    )


def create_summary_statistics(df: pd.DataFrame) -> None:
    """Create summary statistics table using calculated metrics."""
    summary_stats = []

    # Overall statistics
    for metric in METRICS:
        summary_stats.append(
            {
                "category": "Overall",
                "metric": metric,
                "mean": df[metric].mean(),
                "std": df[metric].std(),
                "min": df[metric].min(),
                "max": df[metric].max(),
            }
        )

    # By experiment type
    for exp_type in EXPERIMENT_TYPES:
        data = df[df["experiment_type"] == exp_type]
        if not data.empty:
            for metric in METRICS:
                summary_stats.append(
                    {
                        "category": exp_type,
                        "metric": metric,
                        "mean": data[metric].mean(),
                        "std": data[metric].std(),
                        "min": data[metric].min(),
                        "max": data[metric].max(),
                    }
                )

    summary_df = pd.DataFrame(summary_stats)
    eval_utils.save_results(
        summary_df, {}, str(ANALYSIS_OUTPUT_DIR / "summary_statistics")
    )


def generate_visualization_report() -> None:
    """Generate comprehensive visualization report with clear data preprocessing."""
    df = preprocess_experimental_data()

    create_metrics_by_context_window(df)
    create_metrics_by_concern_count(df)
    create_inference_time_vs_metrics(df)
    create_context_window_vs_metrics(df)
    create_comparison_plots(df)
    create_summary_statistics(df)


if __name__ == "__main__":
    generate_visualization_report()
