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
CONTEXT_WINDOWS = [1024, 2048, 4096, 8192, 12288]
EXPERIMENT_TYPES = ["with_message", "diff_only"]
METRICS = ["accuracy", "f1", "precision", "recall"]

plt.style.use("default")
sns.set_palette("husl")


def load_and_calculate_metrics() -> pd.DataFrame:
    """Load results and calculate all metrics including accuracy."""
    all_data = []

    for exp_type in EXPERIMENT_TYPES:
        for context_window in CONTEXT_WINDOWS:
            # Try multiple model name patterns
            for model_pattern in ["microsoft_phi-4", "gpt-4o-mini", "gpt-4o"]:
                csv_path = (
                    RESULTS_BASE_DIR / exp_type / f"{model_pattern}_{context_window}.csv"
                )
                
                if not csv_path.exists():
                    continue
                
                df = pd.read_csv(csv_path)
                
                # Calculate metrics for each row
                metrics_list = []
                for _, row in df.iterrows():
                    try:
                        # Parse string representations of lists
                        predicted_types = eval(row["predicted_types"]) if isinstance(row["predicted_types"], str) else row["predicted_types"]
                        actual_types = eval(row["actual_types"]) if isinstance(row["actual_types"], str) else row["actual_types"]
                        
                        # Calculate metrics using utils.eval
                        metrics = eval_utils.calculate_metrics(predicted_types, actual_types)
                        
                        metrics_list.append({
                            "model": model_pattern,
                            "context_window": context_window,
                            "experiment_type": exp_type,
                            "predicted_count": len(predicted_types),
                            "actual_count": len(actual_types),
                            "accuracy": float(metrics["exact_match"]),
                            "f1": metrics["f1"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "inference_time": row.get("inference_time", 0.0)
                        })
                    except Exception as e:
                        print(f"Error processing row: {e}")
                        continue
                
                if metrics_list:
                    all_data.extend(metrics_list)

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def create_metrics_by_context_window(df: pd.DataFrame) -> None:
    """Create visualization of metrics by context window."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        
        # Prepare data for plotting
        for exp_type in EXPERIMENT_TYPES:
            data = df[df["experiment_type"] == exp_type]
            if data.empty:
                continue
                
            grouped = data.groupby("context_window")[metric].mean().reset_index()
            
            ax.plot(
                grouped["context_window"],
                grouped[metric],
                marker='o',
                label=exp_type.replace("_", " ").title(),
                linewidth=2,
                markersize=8
            )
        
        ax.set_title(f'{metric.title()} by Context Window', fontsize=14)
        ax.set_xlabel('Context Window Size', fontsize=12)
        ax.set_ylabel(metric.title(), fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        ANALYSIS_OUTPUT_DIR / "metrics_by_context_window.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    # Save as table
    table_data = []
    for exp_type in EXPERIMENT_TYPES:
        for context_window in CONTEXT_WINDOWS:
            data = df[(df["experiment_type"] == exp_type) & (df["context_window"] == context_window)]
            if not data.empty:
                row = {"experiment_type": exp_type, "context_window": context_window}
                for metric in METRICS:
                    row[metric] = data[metric].mean()
                table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    table_df.to_csv(ANALYSIS_OUTPUT_DIR / "metrics_by_context_window_table.csv", index=False)
    print("Saved: metrics_by_context_window_table.csv")


def create_metrics_by_concern_count(df: pd.DataFrame) -> None:
    """Create visualization of metrics by concern count."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        
        # Plot for each experiment type
        for exp_type in EXPERIMENT_TYPES:
            data = df[df["experiment_type"] == exp_type]
            if data.empty:
                continue
                
            # Group by actual concern count
            grouped = data.groupby("actual_count")[metric].mean().reset_index()
            
            ax.plot(
                grouped["actual_count"],
                grouped[metric],
                marker='o',
                label=exp_type.replace("_", " ").title(),
                linewidth=2,
                markersize=8
            )
        
        ax.set_title(f'{metric.title()} by Actual Concern Count', fontsize=14)
        ax.set_xlabel('Number of Concerns', fontsize=12)
        ax.set_ylabel(metric.title(), fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3])
    
    plt.tight_layout()
    plt.savefig(
        ANALYSIS_OUTPUT_DIR / "metrics_by_concern_count.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    # Save as table
    table_data = []
    for exp_type in EXPERIMENT_TYPES:
        for concern_count in [1, 2, 3]:
            data = df[(df["experiment_type"] == exp_type) & (df["actual_count"] == concern_count)]
            if not data.empty:
                row = {"experiment_type": exp_type, "concern_count": concern_count}
                for metric in METRICS:
                    row[metric] = data[metric].mean()
                table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    table_df.to_csv(ANALYSIS_OUTPUT_DIR / "metrics_by_concern_count_table.csv", index=False)
    print("Saved: metrics_by_concern_count_table.csv")


def create_combined_heatmap(df: pd.DataFrame) -> None:
    """Create heatmap showing metrics across context windows and concern counts."""
    for exp_type in EXPERIMENT_TYPES:
        data = df[df["experiment_type"] == exp_type]
        if data.empty:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(METRICS):
            ax = axes[i]
            
            # Create pivot table
            pivot = data.pivot_table(
                values=metric,
                index="actual_count",
                columns="context_window",
                aggfunc='mean'
            )
            
            # Create heatmap
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                ax=ax,
                cbar_kws={'label': metric.title()}
            )
            
            ax.set_title(f'{metric.title()} Heatmap', fontsize=12)
            ax.set_xlabel('Context Window', fontsize=10)
            ax.set_ylabel('Concern Count', fontsize=10)
        
        plt.suptitle(f'Metrics Heatmap - {exp_type.replace("_", " ").title()}', fontsize=16)
        plt.tight_layout()
        plt.savefig(
            ANALYSIS_OUTPUT_DIR / f"metrics_heatmap_{exp_type}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


def create_comparison_plots(df: pd.DataFrame) -> None:
    """Create comparison plots between with_message and diff_only."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Calculate average metrics for each experiment type
    comparison_data = []
    for exp_type in EXPERIMENT_TYPES:
        data = df[df["experiment_type"] == exp_type]
        if not data.empty:
            for metric in METRICS:
                comparison_data.append({
                    "experiment_type": exp_type,
                    "metric": metric,
                    "value": data[metric].mean()
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create grouped bar plot
    pivot = comparison_df.pivot(index="metric", columns="experiment_type", values="value")
    pivot.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('Average Metrics Comparison: With Message vs Diff Only', fontsize=16)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.legend(title='Experiment Type', labels=['Diff Only', 'With Message'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(
        ANALYSIS_OUTPUT_DIR / "metrics_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    # Save comparison table
    pivot.to_csv(ANALYSIS_OUTPUT_DIR / "metrics_comparison_table.csv")
    print("Saved: metrics_comparison_table.csv")


def create_summary_statistics(df: pd.DataFrame) -> None:
    """Create summary statistics table."""
    summary_stats = []
    
    # Overall statistics
    for metric in METRICS:
        summary_stats.append({
            "category": "Overall",
            "metric": metric,
            "mean": df[metric].mean(),
            "std": df[metric].std(),
            "min": df[metric].min(),
            "max": df[metric].max()
        })
    
    # By experiment type
    for exp_type in EXPERIMENT_TYPES:
        data = df[df["experiment_type"] == exp_type]
        if not data.empty:
            for metric in METRICS:
                summary_stats.append({
                    "category": exp_type,
                    "metric": metric,
                    "mean": data[metric].mean(),
                    "std": data[metric].std(),
                    "min": data[metric].min(),
                    "max": data[metric].max()
                })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(ANALYSIS_OUTPUT_DIR / "summary_statistics.csv", index=False)
    print("Saved: summary_statistics.csv")
    
    # Print summary to console
    print("\n=== Summary Statistics ===")
    print(summary_df.to_string(index=False))


def generate_visualization_report() -> None:
    """Generate comprehensive visualization report with tables and plots."""
    print("Loading experimental results and calculating metrics...")
    df = load_and_calculate_metrics()
    
    if df.empty:
        print("No experimental data found in results directory.")
        return
    
    print(f"Loaded {len(df)} experimental results")
    print(f"Models found: {df['model'].unique().tolist()}")
    print(f"Experiment types: {df['experiment_type'].unique().tolist()}")
    print(f"Context windows: {sorted(df['context_window'].unique().tolist())}")
    
    # Create all visualizations
    print("\nGenerating visualizations...")
    
    print("1. Creating metrics by context window...")
    create_metrics_by_context_window(df)
    
    print("2. Creating metrics by concern count...")
    create_metrics_by_concern_count(df)
    
    print("3. Creating combined heatmaps...")
    create_combined_heatmap(df)
    
    print("4. Creating comparison plots...")
    create_comparison_plots(df)
    
    print("5. Creating summary statistics...")
    create_summary_statistics(df)
    
    print("\n=== Visualization Report Complete! ===")
    print(f"All visualizations and tables saved to: {ANALYSIS_OUTPUT_DIR}")
    print("\nGenerated files:")
    print("Plots:")
    print("  - metrics_by_context_window.png")
    print("  - metrics_by_concern_count.png")
    print("  - metrics_heatmap_with_message.png")
    print("  - metrics_heatmap_diff_only.png")
    print("  - metrics_comparison.png")
    print("\nTables:")
    print("  - metrics_by_context_window_table.csv")
    print("  - metrics_by_concern_count_table.csv")
    print("  - metrics_comparison_table.csv")
    print("  - summary_statistics.csv")


if __name__ == "__main__":
    generate_visualization_report()
