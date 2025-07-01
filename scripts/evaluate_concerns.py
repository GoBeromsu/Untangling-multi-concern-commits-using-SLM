#!/usr/bin/env python3
"""
Concern Classification Evaluation Script

This script evaluates concern classification models on generated datasets
and outputs results in markdown table format with real-time updates.
"""

import argparse
import json
import os
import glob
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import pandas as pd
from dotenv import load_dotenv

# Add src to path to import modules
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

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
        print(f"Error loading concern test dataset: {str(e)}")
        return [], {}


def calculate_case_metrics(
    predicted_types: List[str], actual_types: List[str]
) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 for a single case using strict multiset matching."""
    predicted_counter = Counter(predicted_types)
    actual_counter = Counter(actual_types)

    # STRICT MULTISET MATCHING: Calculate TP, FP, FN with exact count requirements
    all_types = set(predicted_counter.keys()) | set(actual_counter.keys())
    tp = sum(min(predicted_counter[t], actual_counter[t]) for t in all_types)
    fp = sum(max(0, predicted_counter[t] - actual_counter[t]) for t in all_types)
    fn = sum(max(0, actual_counter[t] - predicted_counter[t]) for t in all_types)

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def classify_concern_composition(types: List[str]) -> str:
    """Classify the concern composition based on the types involved."""
    purpose_types = {"feat", "fix", "style", "refactor"}
    object_types = {"docs", "test", "cicd", "build"}

    purpose_count = sum(1 for t in types if t in purpose_types)
    object_count = sum(1 for t in types if t in object_types)

    if len(types) == 1:
        if types[0] in purpose_types:
            return "purpose"
        else:
            return "object"
    elif len(types) == 2:
        if purpose_count == 2:
            return "purpose + purpose"
        elif object_count == 2:
            return "object + object"
        elif purpose_count == 1 and object_count == 1:
            # Check order: if object comes first, then "object + purpose"
            if types[0] in object_types:
                return "object + purpose"
            else:
                return "purpose + object"
        else:
            return "mixed"
    elif len(types) == 3:
        if purpose_count == 3:
            return "purpose x 3"
        elif object_count == 3:
            return "object x 3"
        elif purpose_count == 2 and object_count == 1:
            return "purpose x 2 + object"
        elif purpose_count == 1 and object_count == 2:
            return "purpose + object x 2"
        else:
            return "mixed"
    else:
        return "complex"


def get_composition_order(composition: str) -> int:
    """Return sort order for composition types."""
    order_map = {
        "object": 1,
        "purpose": 2,
        "object + object": 3,
        "object + purpose": 4,
        "purpose + purpose": 5,
        "purpose x 3": 6,
        "object x 3": 7,
        "purpose x 2 + object": 8,
        "purpose + object x 2": 9,
    }
    return order_map.get(composition, 99)


def update_markdown_file(
    evaluation_results: List[Dict[str, Any]], output_file: Path, is_final: bool = False
) -> None:
    """Update the markdown file with current results."""
    table_rows = []

    for result in evaluation_results:
        if "error" in result:
            continue

        metadata = result.get("metadata", {})
        types = metadata.get("types", [])
        allow_same_types = metadata.get("allow_same_types", False)

        # Generate type display name
        if len(types) == 1:
            type_display = types[0]
        else:
            type_suffix = " (m)" if allow_same_types else " (u)"
            type_display = " + ".join(types) + type_suffix

        # Classify concern composition
        composition = classify_concern_composition(types)

        # Format metrics as percentages
        precision = f"{result['precision']:.1%}"
        recall = f"{result['recall']:.1%}"
        f1_macro = f"{result['f1_macro']:.1%}"
        count_accuracy = f"{result['count_accuracy']:.1%}"

        table_rows.append(
            {
                "Type": type_display,
                "Precision": precision,
                "Recall": recall,
                "Type F1 (Macro)": f1_macro,
                "Count Accuracy": count_accuracy,
                "Concern Composition": composition,
                "order": get_composition_order(composition),
            }
        )

    # Sort by composition order and then by type
    table_rows.sort(key=lambda x: (x["order"], x["Type"]))

    # Generate markdown content
    if not table_rows:
        content = "No valid results to display yet.\n"
    else:
        headers = [
            "Type",
            "Precision",
            "Recall",
            "Type F1 (Macro)",
            "Count Accuracy",
            "Concern Composition",
        ]

        # Create header
        content = f"# Concern Classification Evaluation Results\n\n"
        if not is_final:
            content += (
                f"**Status**: Evaluation in progress... ({len(table_rows)} completed)\n"
            )
            content += f"**Last updated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        else:
            content += f"**Status**: Evaluation completed ✅\n"
            content += f"**Total evaluations**: {len(table_rows)}\n"
            content += f"**Completed at**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        content += "| " + " | ".join(headers) + " |\n"
        content += "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|\n"

        # Add rows
        for row in table_rows:
            values = [row[h] for h in headers]
            content += "| " + " | ".join(values) + " |\n"

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)


def evaluate_dataset(
    api_key: str, dataset_path: str, system_prompt: str
) -> Dict[str, Any]:
    """Evaluate a single dataset and return metrics."""
    test_dataset, metadata = load_concern_test_dataset(dataset_path)

    if not test_dataset:
        return {"error": f"Failed to load dataset: {dataset_path}"}

    print(f"  📊 Evaluating {len(test_dataset)} cases...")

    results = []
    for i, test_case in enumerate(test_dataset):
        print(f"    Case {i+1}/{len(test_dataset)}", end="\r")

        diff = test_case.get("tangleChange", "")
        atomic_changes = test_case.get("atomicChanges", [])
        actual_types = [change.get("label", "") for change in atomic_changes]

        # Get model prediction
        model_response = openai_api_call(api_key, diff, system_prompt)

        try:
            prediction_data = json.loads(model_response)
            predicted_types = prediction_data.get("types", [])
            predicted_count = prediction_data.get("count", 0)
        except json.JSONDecodeError:
            predicted_types = []
            predicted_count = 0

        # Calculate case metrics
        precision, recall, f1 = calculate_case_metrics(predicted_types, actual_types)
        count_match = predicted_count == len(actual_types)

        results.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "count_match": count_match,
                "predicted_count": predicted_count,
                "actual_count": len(actual_types),
            }
        )

    print()  # Clear the progress line

    # Calculate aggregate metrics
    total_cases = len(results)
    avg_precision = (
        sum(r["precision"] for r in results) / total_cases if total_cases > 0 else 0.0
    )
    avg_recall = (
        sum(r["recall"] for r in results) / total_cases if total_cases > 0 else 0.0
    )
    avg_f1 = sum(r["f1"] for r in results) / total_cases if total_cases > 0 else 0.0
    count_accuracy = (
        sum(1 for r in results if r["count_match"]) / total_cases
        if total_cases > 0
        else 0.0
    )

    return {
        "dataset_path": dataset_path,
        "metadata": metadata,
        "total_cases": total_cases,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_macro": avg_f1,
        "count_accuracy": count_accuracy,
        "results": results,
    }


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate concern classification datasets and generate markdown results with real-time updates"
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets/tangled/codefuse-hqcm"),
        help="Directory containing dataset JSON files",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output file for markdown results (optional, prints to stdout if not provided)",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for OpenAI API key",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv(args.api_key_env)

    if not api_key:
        print(f"❌ Error: {args.api_key_env} environment variable not found")
        return

    print("✅ API key loaded successfully")

    # Find all dataset files
    dataset_files = list(args.datasets_dir.glob("*.json"))

    if not dataset_files:
        print(f"❌ No dataset files found in {args.datasets_dir}")
        return

    print(f"📁 Found {len(dataset_files)} dataset files")

    # Get system prompt
    system_prompt = get_system_prompt()

    # Set default output file if not provided
    if not args.output_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_file = Path(f"results/concern_evaluation_{timestamp}.md")
        args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize with empty results
    evaluation_results = []
    update_markdown_file(evaluation_results, args.output_file, is_final=False)
    print(f"📄 Real-time results: {args.output_file}")

    # Evaluate all datasets with real-time updates
    for i, dataset_file in enumerate(sorted(dataset_files)):
        print(f"\n🔄 [{i+1}/{len(dataset_files)}] Evaluating: {dataset_file.name}")
        result = evaluate_dataset(api_key, str(dataset_file), system_prompt)
        evaluation_results.append(result)

        # Update markdown file after each dataset
        update_markdown_file(evaluation_results, args.output_file, is_final=False)
        print(
            f"  ✅ Updated results file: {len(evaluation_results)} evaluations completed"
        )

    # Final update
    print("\n📊 Generating final results table...")
    update_markdown_file(evaluation_results, args.output_file, is_final=True)

    print(f"✅ Final results saved to: {args.output_file}")
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    print(
        f"📂 Total datasets evaluated: {len([r for r in evaluation_results if 'error' not in r])}"
    )
    print(f"📄 Results file: {args.output_file}")


if __name__ == "__main__":
    main()
