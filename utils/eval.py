"""Evaluation utilities for parsing outputs and calculating metrics."""

from typing import Dict, Any, Set
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support


def load_dataset(dataset_split: str) -> pd.DataFrame:
    """Load dataset from local CSV file."""
    if dataset_split == "test":
        csv_path = Path("../datasets/data/tangled_ccs_dataset_test.csv")
    elif dataset_split == "train":
        csv_path = Path("../datasets/data/tangled_ccs_dataset_train.csv")
    else:
        csv_path = Path("../datasets/data/tangled_ccs_dataset.csv")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    return df


def parse_model_output(output_str: str) -> Set[str]:
    """Parse model output text to extract concerns as a set."""
    concerns = set()

    # Try to find JSON-like structures first
    json_pattern = r"\[([^\]]+)\]"
    matches = re.findall(json_pattern, output_str)

    if matches:
        for match in matches:
            items = [item.strip().strip("\"'") for item in match.split(",")]
            concerns.update(item for item in items if item)
    else:
        # Fallback: look for numbered lists or bullet points
        lines = output_str.split("\n")
        for line in lines:
            line = line.strip()
            if re.match(r"^[\d\-\*\•]\s*\.?\s*", line):
                concern = re.sub(r"^[\d\-\*\•]\s*\.?\s*", "", line).strip()
                if concern:
                    concerns.add(concern)

    return concerns


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate F1, Precision, Recall metrics from predictions DataFrame."""
    # Convert sets to binary vectors for each unique concern
    all_concerns = set()
    for pred_set in df["predictions"]:
        all_concerns.update(pred_set)
    for gt_set in df["ground_truth"]:
        all_concerns.update(gt_set)

    all_concerns = sorted(list(all_concerns))

    # Create binary matrices
    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        true_vector = [
            1 if concern in row["ground_truth"] else 0 for concern in all_concerns
        ]
        pred_vector = [
            1 if concern in row["predictions"] else 0 for concern in all_concerns
        ]
        y_true.append(true_vector)
        y_pred.append(pred_vector)

    # Calculate metrics using sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    return {
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def save_metric_csvs(
    model_name: str, metrics: Dict[str, float], output_dir: str
) -> None:
    """Save metrics for a single model to CSV files in real-time."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # File paths
    f1_path = output_path / "macro_f1.csv"
    precision_path = output_path / "macro_precision.csv"
    recall_path = output_path / "macro_recall.csv"

    # Read existing data or create new
    try:
        f1_df = pd.read_csv(f1_path)
        precision_df = pd.read_csv(precision_path)
        recall_df = pd.read_csv(recall_path)
    except FileNotFoundError:
        f1_df = pd.DataFrame()
        precision_df = pd.DataFrame()
        recall_df = pd.DataFrame()

    # Add new model data
    f1_df[model_name] = [metrics["f1_score"]]
    precision_df[model_name] = [metrics["precision"]]
    recall_df[model_name] = [metrics["recall"]]

    # Save updated CSV files
    f1_df.to_csv(f1_path, index=False)
    precision_df.to_csv(precision_path, index=False)
    recall_df.to_csv(recall_path, index=False)


def save_results(df: pd.DataFrame, metrics: Dict[str, float], output_dir: str) -> None:
    """Save DataFrame as predictions.csv and metrics as metrics.json."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / "predictions.csv", index=False)

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def plot_graph(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: str,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
) -> None:
    """Create and save a line plot for analysis."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, marker="o")

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
