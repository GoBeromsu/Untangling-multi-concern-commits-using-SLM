#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

RESULTS_BASE_DIR = Path("results")
ANALYSIS_OUTPUT_DIR = Path("results/analysis")
CONTEXT_WINDOWS = [1024, 2048, 4096, 8192, 12288]
EXPERIMENT_TYPES = ["with_message", "diff_only"]

plt.style.use("default")
sns.set_palette("husl")


def load_results_data() -> pd.DataFrame:
    """Load and combine all experimental results from results directory."""
    all_data = []

    for exp_type in EXPERIMENT_TYPES:
        for context_window in CONTEXT_WINDOWS:
            csv_path = (
                RESULTS_BASE_DIR / exp_type / f"microsoft_phi-4_{context_window}.csv"
            )

            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)

            # Add experiment metadata
            df["context_window"] = context_window
            df["experiment_type"] = exp_type

            all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def create_metrics_by_concern_count_visualization(df: pd.DataFrame) -> None:
    """Create comprehensive visualization of metrics by concern count."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    metrics = ["f1", "precision", "recall"]

    for i, exp_type in enumerate(["with_message", "diff_only"]):
        data = df[df["experiment_type"] == exp_type]

        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            count_combinations = []
            metric_values = []
            labels = []

            for pred_count in [1, 2, 3]:
                for actual_count in [1, 2, 3]:
                    subset = data[
                        (data["predicted_count"] == pred_count)
                        & (data["actual_count"] == actual_count)
                    ]

                    if len(subset) > 0:
                        count_combinations.append((pred_count, actual_count))
                        metric_values.append(subset[metric].mean())
                        labels.append(f"P{pred_count}/A{actual_count}")

            if metric_values:
                x_pos = np.arange(len(labels))
                bars = ax.bar(x_pos, metric_values)

                for idx, (pred, actual) in enumerate(count_combinations):
                    bars[idx].set_color(
                        "lightgreen" if pred == actual else "lightcoral"
                    )

                ax.set_title(
                    f'{metric.title()} by Predicted/Actual Count\n({exp_type.replace("_", " ").title()})'
                )
                ax.set_xlabel("Predicted Count / Actual Count")
                ax.set_ylabel(metric.title())
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45)
                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        ANALYSIS_OUTPUT_DIR / "metrics_by_concern_count_fixed.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_separated_concern_count_visualization(df: pd.DataFrame) -> None:
    """Create visualization showing actual vs predicted concern counts separately."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, exp_type in enumerate(["with_message", "diff_only"]):
        data = df[df["experiment_type"] == exp_type]

        actual_f1 = data.groupby("actual_count")["f1"].mean()
        axes[i, 0].bar(actual_f1.index, actual_f1.values, color="lightblue")
        axes[i, 0].set_title(
            f'F1 by Actual Concern Count\n({exp_type.replace("_", " ").title()})'
        )
        axes[i, 0].set_xlabel("Actual Concern Count")
        axes[i, 0].set_ylabel("F1 Score")
        axes[i, 0].grid(True, alpha=0.3)

        predicted_f1 = data.groupby("predicted_count")["f1"].mean()
        axes[i, 1].bar(predicted_f1.index, predicted_f1.values, color="lightgreen")
        axes[i, 1].set_title(
            f'F1 by Predicted Concern Count\n({exp_type.replace("_", " ").title()})'
        )
        axes[i, 1].set_xlabel("Predicted Concern Count")
        axes[i, 1].set_ylabel("F1 Score")
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        ANALYSIS_OUTPUT_DIR / "metrics_by_concern_count_separate.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def generate_visualization_report() -> None:
    """Generate comprehensive visualization report and save results."""
    print("Loading experimental results from results directory...")
    df = load_results_data()

    if df.empty:
        print("No experimental data found in results directory.")
        return

    print(f"Loaded {len(df)} experimental results")
    print(f"Experiment types: {df['experiment_type'].unique().tolist()}")
    print(f"Context windows: {sorted(df['context_window'].unique().tolist())}")

    print("Creating metrics by concern count visualization...")
    create_metrics_by_concern_count_visualization(df)

    print("Creating separated concern count visualization...")
    create_separated_concern_count_visualization(df)

    print("\nVisualization Report Complete!")
    print("Generated visualizations:")
    print("- metrics_by_concern_count_fixed.png: Predicted/Actual combinations")
    print("- metrics_by_concern_count_separate.png: Separate actual vs predicted")
    print(f"All visualizations saved to: {ANALYSIS_OUTPUT_DIR}")


if __name__ == "__main__":
    generate_visualization_report()
