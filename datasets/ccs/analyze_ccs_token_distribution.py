#!/usr/bin/env python3
"""
Analyze CCS dataset diff sizes by commit type using token count.
Purpose: Find types with smaller diffs for LLM input using precise token measurements.
"""

import pandas as pd
import numpy as np
import tiktoken
from typing import Dict, List, Tuple
import os


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to character count / 4 if tiktoken fails
        return len(text) // 4


def calculate_token_distribution(
    df: pd.DataFrame, type_column: str, diff_column: str
) -> Dict[str, Dict]:
    """Calculate token-based statistics and distribution by type."""

    # Define token ranges
    ranges = [
        (0, 1000, "≤1K"),
        (1001, 2000, "1K-2K"),
        (2001, 3000, "2K-3K"),
        (3001, 4000, "3K-4K"),
        (4001, 5000, "4K-5K"),
        (5001, float("inf"), ">5K"),
    ]

    stats_by_type = {}

    for commit_type in df[type_column].unique():
        type_data = df[df[type_column] == commit_type][diff_column]

        print(f"Processing {commit_type}: {len(type_data)} samples...")

        # Calculate token counts for all diffs of this type
        token_counts = []
        for idx, diff_text in enumerate(type_data):
            if idx % 20 == 0:  # Progress indicator
                print(f"  Processing {idx+1}/{len(type_data)}")

            if pd.isna(diff_text):
                token_count = 0
            else:
                token_count = count_tokens(str(diff_text))
            token_counts.append(token_count)

        token_counts = np.array(token_counts)

        # Calculate distribution by ranges
        distribution = {}
        for min_val, max_val, label in ranges:
            if max_val == float("inf"):
                count = np.sum(token_counts > min_val)
            else:
                count = np.sum((token_counts >= min_val) & (token_counts <= max_val))

            percentage = (count / len(token_counts)) * 100
            distribution[label] = {"count": int(count), "percentage": percentage}

        # Calculate overall statistics
        stats_by_type[commit_type] = {
            "total_count": len(token_counts),
            "mean": float(np.mean(token_counts)),
            "std": float(np.std(token_counts)),
            "min": int(np.min(token_counts)),
            "max": int(np.max(token_counts)),
            "median": float(np.median(token_counts)),
            "q25": float(np.percentile(token_counts, 25)),
            "q75": float(np.percentile(token_counts, 75)),
            "q90": float(np.percentile(token_counts, 90)),
            "q95": float(np.percentile(token_counts, 95)),
            "q99": float(np.percentile(token_counts, 99)),
            "distribution": distribution,
        }

    return stats_by_type


def generate_token_markdown_report(stats: Dict[str, Dict], output_file: str) -> None:
    """Generate comprehensive markdown report with token-based analysis."""

    # Sort types by median token count (ascending)
    sorted_types = sorted(stats.keys(), key=lambda x: stats[x]["median"])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# CCS Dataset: Token-Based Diff Size Analysis by Commit Type\n\n")
        f.write("## Purpose\n")
        f.write(
            "Analysis of git diff sizes using **token count** (via tiktoken) by commit type to identify types with smaller diffs suitable for LLM input.\n\n"
        )

        f.write(
            "## Summary: Types Ranked by Median Token Count (Smallest to Largest)\n\n"
        )
        f.write(
            "| Rank | Type | Count | Median Tokens | Mean Tokens | 95th Percentile |\n"
        )
        f.write(
            "|------|------|-------|---------------|-------------|----------------|\n"
        )

        for rank, commit_type in enumerate(sorted_types, 1):
            s = stats[commit_type]
            f.write(
                f"| {rank} | **{commit_type}** | {s['total_count']} | {s['median']:,.0f} | {s['mean']:,.0f} | {s['q95']:,.0f} |\n"
            )

        f.write("\n## Token Range Distribution by Type\n\n")
        f.write(
            "Distribution of diffs across token count ranges for each commit type:\n\n"
        )

        # Create distribution table
        ranges = ["≤1K", "1K-2K", "2K-3K", "3K-4K", "4K-5K", ">5K"]
        f.write("| Type | ≤1K | 1K-2K | 2K-3K | 3K-4K | 4K-5K | >5K |\n")
        f.write("|------|-----|-------|-------|-------|-------|-----|\n")

        for commit_type in sorted_types:
            s = stats[commit_type]
            row = f"| **{commit_type}** |"
            for range_label in ranges:
                dist = s["distribution"][range_label]
                row += f" {dist['count']} ({dist['percentage']:.1f}%) |"
            f.write(row + "\n")

        f.write("\n## Recommendations for LLM Input (Token-Based)\n\n")

        # Categorize types based on token counts
        excellent_types = [
            t for t in sorted_types if stats[t]["q95"] < 2000
        ]  # 95th percentile < 2K tokens
        good_types = [
            t for t in sorted_types if 2000 <= stats[t]["q95"] < 4000
        ]  # 2K-4K tokens
        moderate_types = [
            t for t in sorted_types if 4000 <= stats[t]["q95"] < 8000
        ]  # 4K-8K tokens
        challenging_types = [
            t for t in sorted_types if stats[t]["q95"] >= 8000
        ]  # ≥8K tokens

        if excellent_types:
            f.write(
                "### 🌟 **Excellent for LLM Input** (95th percentile < 2,000 tokens)\n"
            )
            for commit_type in excellent_types:
                s = stats[commit_type]
                small_count = sum(
                    s["distribution"][r]["count"] for r in ["≤1K", "1K-2K"]
                )
                small_pct = (small_count / s["total_count"]) * 100
                f.write(
                    f"- **{commit_type}**: median {s['median']:,.0f} tokens, 95th percentile {s['q95']:,.0f} tokens, {small_pct:.1f}% ≤2K tokens\n"
                )

        if good_types:
            f.write("\n### ✅ **Good for LLM Input** (95th percentile 2K-4K tokens)\n")
            for commit_type in good_types:
                s = stats[commit_type]
                small_count = sum(
                    s["distribution"][r]["count"] for r in ["≤1K", "1K-2K", "2K-3K"]
                )
                small_pct = (small_count / s["total_count"]) * 100
                f.write(
                    f"- **{commit_type}**: median {s['median']:,.0f} tokens, 95th percentile {s['q95']:,.0f} tokens, {small_pct:.1f}% ≤3K tokens\n"
                )

        if moderate_types:
            f.write(
                "\n### ⚠️ **Moderate for LLM Input** (95th percentile 4K-8K tokens)\n"
            )
            for commit_type in moderate_types:
                s = stats[commit_type]
                large_count = s["distribution"][">5K"]["count"]
                large_pct = (large_count / s["total_count"]) * 100
                f.write(
                    f"- **{commit_type}**: median {s['median']:,.0f} tokens, 95th percentile {s['q95']:,.0f} tokens, {large_pct:.1f}% >5K tokens\n"
                )

        if challenging_types:
            f.write(
                "\n### ❌ **Challenging for LLM Input** (95th percentile ≥ 8K tokens)\n"
            )
            for commit_type in challenging_types:
                s = stats[commit_type]
                large_count = s["distribution"][">5K"]["count"]
                large_pct = (large_count / s["total_count"]) * 100
                f.write(
                    f"- **{commit_type}**: median {s['median']:,.0f} tokens, 95th percentile {s['q95']:,.0f} tokens, {large_pct:.1f}% >5K tokens\n"
                )

        f.write("\n## Detailed Statistics by Type\n\n")

        for commit_type in sorted_types:
            s = stats[commit_type]
            f.write(f"### {commit_type}\n")
            f.write(f"- **Total Count**: {s['total_count']} samples\n")
            f.write(f"- **Mean**: {s['mean']:,.0f} tokens\n")
            f.write(f"- **Median**: {s['median']:,.0f} tokens\n")
            f.write(f"- **Standard Deviation**: {s['std']:,.0f}\n")
            f.write(f"- **Range**: {s['min']:,.0f} - {s['max']:,.0f} tokens\n")
            f.write(f"- **Quartiles**: Q25={s['q25']:,.0f}, Q75={s['q75']:,.0f}\n")
            f.write(
                f"- **Percentiles**: 90th={s['q90']:,.0f}, 95th={s['q95']:,.0f}, 99th={s['q99']:,.0f}\n"
            )

            f.write("\n**Token Range Distribution:**\n")
            for range_label, dist_data in s["distribution"].items():
                f.write(
                    f"  - {range_label}: {dist_data['count']} samples ({dist_data['percentage']:.1f}%)\n"
                )
            f.write("\n")


def main() -> None:
    """Main analysis function."""
    csv_file = "datasets/ccs/CCS Dataset Training Data.csv"
    output_file = "ccs_token_distribution_analysis.md"

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return

    print("Loading CCS dataset...")
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records")

        type_column = "annotated_type"
        diff_column = "git_diff"

        print(f"Using type column: {type_column}")
        print(f"Using diff column: {diff_column}")
        print("Note: Using tiktoken cl100k_base encoding for token counting\n")

        # Calculate token-based statistics
        print("Calculating token-based statistics and distributions...")
        stats = calculate_token_distribution(df, type_column, diff_column)

        # Generate markdown report
        print(f"\nGenerating markdown report: {output_file}")
        generate_token_markdown_report(stats, output_file)

        print(f"Analysis complete! Report saved to {output_file}")

        # Print quick summary
        print("\nQuick Summary (sorted by median token count):")
        sorted_types = sorted(stats.keys(), key=lambda x: stats[x]["median"])
        for commit_type in sorted_types:
            s = stats[commit_type]
            small_count = sum(s["distribution"][r]["count"] for r in ["≤1K", "1K-2K"])
            print(
                f"  {commit_type}: median {s['median']:,.0f} tokens, {small_count}/{s['total_count']} (≤2K tokens)"
            )

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
