#!/usr/bin/env python3
"""
Analyze git diff token distribution by commit type and type combinations.
Purpose: Find optimal types and combinations for LLM input using precise token measurements.
"""

import pandas as pd
import numpy as np
import tiktoken
from typing import Dict, List, Tuple, Set
from pathlib import Path
import json
import argparse


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to character count / 4 if tiktoken fails
        return len(text) // 4


def parse_types_from_row(types_value: str) -> List[str]:
    """Parse commit types from a row value (handles both single types and JSON arrays)."""
    if pd.isna(types_value):
        return []

    types_str = str(types_value).strip()
    if types_str.startswith("["):
        try:
            return json.loads(types_str)
        except json.JSONDecodeError:
            return [types_str]
    else:
        return [types_str]


def create_expanded_dataset(
    df: pd.DataFrame, type_column: str, diff_column: str
) -> pd.DataFrame:
    """Create expanded dataset with individual and combination type analysis."""
    expanded_data = []

    for _, row in df.iterrows():
        diff_text = row[diff_column]
        types_list = parse_types_from_row(row[type_column])

        if not types_list:
            continue

        # Add individual types
        for commit_type in types_list:
            expanded_data.append(
                {
                    "type": commit_type,
                    "type_combination": tuple(sorted(types_list)),
                    "combination_size": len(types_list),
                    "diff": diff_text,
                }
            )

    return pd.DataFrame(expanded_data)


def calculate_token_ranges() -> List[Tuple[int, int, str]]:
    """Define token ranges for analysis."""
    return [
        (0, 1000, "≤1K"),
        (1001, 2000, "1K-2K"),
        (2001, 3000, "2K-3K"),
        (3001, 4000, "3K-4K"),
        (4001, 5000, "4K-5K"),
        (5001, float("inf"), ">5K"),
    ]


def calculate_token_statistics(
    token_counts: np.ndarray, ranges: List[Tuple[int, int, str]]
) -> Dict:
    """Calculate comprehensive token statistics and distribution."""
    # Calculate distribution by ranges
    distribution = {}
    for min_val, max_val, label in ranges:
        if max_val == float("inf"):
            count = np.sum(token_counts > min_val)
        else:
            count = np.sum((token_counts >= min_val) & (token_counts <= max_val))

        percentage = (count / len(token_counts)) * 100
        distribution[label] = {"count": int(count), "percentage": percentage}

    return {
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


def analyze_individual_types(expanded_df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze token distribution for individual commit types."""
    print("Analyzing individual commit types...")
    ranges = calculate_token_ranges()
    stats_by_type = {}

    for commit_type in sorted(expanded_df["type"].unique()):
        type_data = expanded_df[expanded_df["type"] == commit_type]["diff"]
        print(f"  Processing {commit_type}: {len(type_data)} samples...")

        # Calculate token counts
        token_counts = []
        for diff_text in type_data:
            if pd.isna(diff_text):
                token_count = 0
            else:
                token_count = count_tokens(str(diff_text))
            token_counts.append(token_count)

        token_counts = np.array(token_counts)
        stats_by_type[commit_type] = calculate_token_statistics(token_counts, ranges)

    return stats_by_type


def analyze_type_combinations(expanded_df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze token distribution for type combinations."""
    print("Analyzing type combinations...")
    ranges = calculate_token_ranges()
    stats_by_combination = {}

    # Group by unique combinations
    combination_groups = expanded_df.groupby("type_combination")

    for combination, group in combination_groups:
        if len(combination) == 1:
            continue  # Skip individual types (already analyzed)

        combination_str = "+".join(combination)
        combination_size = len(combination)

        # Get unique diffs for this combination (avoid duplicates)
        unique_diffs = group["diff"].drop_duplicates()

        print(
            f"  Processing {combination_str} (size {combination_size}): {len(unique_diffs)} samples..."
        )

        # Calculate token counts
        token_counts = []
        for diff_text in unique_diffs:
            if pd.isna(diff_text):
                token_count = 0
            else:
                token_count = count_tokens(str(diff_text))
            token_counts.append(token_count)

        if len(token_counts) > 0:
            token_counts = np.array(token_counts)
            stats_by_combination[combination_str] = calculate_token_statistics(
                token_counts, ranges
            )
            stats_by_combination[combination_str]["combination_size"] = combination_size

    return stats_by_combination


def generate_summary_report(
    individual_stats: Dict, combination_stats: Dict, output_file: Path
) -> None:
    """Generate comprehensive markdown report with individual and combination analysis."""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Git Diff Token Distribution Analysis\n\n")
        f.write("## Purpose\n")
        f.write(
            "Analysis of git diff sizes using **token count** (via tiktoken) by commit type and type combinations.\n\n"
        )

        # Individual types summary
        f.write("## Individual Types Summary (Ranked by Median Token Count)\n\n")
        sorted_types = sorted(
            individual_stats.keys(), key=lambda x: individual_stats[x]["median"]
        )

        f.write(
            "| Rank | Type | Count | Median | Mean | 95th Percentile | ≤2K Tokens |\n"
        )
        f.write(
            "|------|------|-------|--------|------|----------------|-------------|\n"
        )

        for rank, commit_type in enumerate(sorted_types, 1):
            s = individual_stats[commit_type]
            small_count = sum(s["distribution"][r]["count"] for r in ["≤1K", "1K-2K"])
            small_pct = (small_count / s["total_count"]) * 100
            f.write(
                f"| {rank} | **{commit_type}** | {s['total_count']} | {s['median']:,.0f} | {s['mean']:,.0f} | {s['q95']:,.0f} | {small_pct:.1f}% |\n"
            )

        # Type combinations summary
        if combination_stats:
            f.write("\n## Type Combinations Summary (Ranked by Median Token Count)\n\n")
            sorted_combinations = sorted(
                combination_stats.keys(), key=lambda x: combination_stats[x]["median"]
            )

            f.write(
                "| Rank | Combination | Size | Count | Median | Mean | 95th Percentile | ≤2K Tokens |\n"
            )
            f.write(
                "|------|-------------|------|-------|--------|------|----------------|-------------|\n"
            )

            for rank, combination in enumerate(sorted_combinations, 1):
                s = combination_stats[combination]
                small_count = sum(
                    s["distribution"][r]["count"] for r in ["≤1K", "1K-2K"]
                )
                small_pct = (small_count / s["total_count"]) * 100
                f.write(
                    f"| {rank} | **{combination}** | {s['combination_size']} | {s['total_count']} | {s['median']:,.0f} | {s['mean']:,.0f} | {s['q95']:,.0f} | {small_pct:.1f}% |\n"
                )

        # Recommendations
        f.write("\n## Recommendations for LLM Input\n\n")

        # Individual types recommendations
        excellent_types = [t for t in sorted_types if individual_stats[t]["q95"] < 2000]
        good_types = [
            t for t in sorted_types if 2000 <= individual_stats[t]["q95"] < 4000
        ]

        if excellent_types:
            f.write(
                "### 🌟 **Excellent Individual Types** (95th percentile < 2K tokens)\n"
            )
            for commit_type in excellent_types:
                s = individual_stats[commit_type]
                small_count = sum(
                    s["distribution"][r]["count"] for r in ["≤1K", "1K-2K"]
                )
                small_pct = (small_count / s["total_count"]) * 100
                f.write(
                    f"- **{commit_type}**: {small_pct:.1f}% ≤2K tokens (median: {s['median']:,.0f})\n"
                )

        if good_types:
            f.write(
                "\n### ✅ **Good Individual Types** (95th percentile 2K-4K tokens)\n"
            )
            for commit_type in good_types:
                s = individual_stats[commit_type]
                small_count = sum(
                    s["distribution"][r]["count"] for r in ["≤1K", "1K-2K", "2K-3K"]
                )
                small_pct = (small_count / s["total_count"]) * 100
                f.write(
                    f"- **{commit_type}**: {small_pct:.1f}% ≤3K tokens (median: {s['median']:,.0f})\n"
                )

        # Combination recommendations
        if combination_stats:
            excellent_combinations = [
                c for c in sorted_combinations if combination_stats[c]["q95"] < 2000
            ]
            good_combinations = [
                c
                for c in sorted_combinations
                if 2000 <= combination_stats[c]["q95"] < 4000
            ]

            if excellent_combinations:
                f.write(
                    "\n### 🌟 **Excellent Type Combinations** (95th percentile < 2K tokens)\n"
                )
                for combination in excellent_combinations:
                    s = combination_stats[combination]
                    small_count = sum(
                        s["distribution"][r]["count"] for r in ["≤1K", "1K-2K"]
                    )
                    small_pct = (small_count / s["total_count"]) * 100
                    f.write(
                        f"- **{combination}**: {small_pct:.1f}% ≤2K tokens (median: {s['median']:,.0f})\n"
                    )

            if good_combinations:
                f.write(
                    "\n### ✅ **Good Type Combinations** (95th percentile 2K-4K tokens)\n"
                )
                for combination in good_combinations:
                    s = combination_stats[combination]
                    small_count = sum(
                        s["distribution"][r]["count"] for r in ["≤1K", "1K-2K", "2K-3K"]
                    )
                    small_pct = (small_count / s["total_count"]) * 100
                    f.write(
                        f"- **{combination}**: {small_pct:.1f}% ≤3K tokens (median: {s['median']:,.0f})\n"
                    )


def print_analysis_summary(individual_stats: Dict, combination_stats: Dict) -> None:
    """Print analysis summary to console."""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print("\nIndividual Types (sorted by median token count):")
    sorted_types = sorted(
        individual_stats.keys(), key=lambda x: individual_stats[x]["median"]
    )
    for commit_type in sorted_types:
        s = individual_stats[commit_type]
        small_count = sum(s["distribution"][r]["count"] for r in ["≤1K", "1K-2K"])
        print(
            f"  {commit_type:12}: median {s['median']:>5,.0f} tokens, {small_count:>3}/{s['total_count']} (≤2K tokens)"
        )

    if combination_stats:
        print("\nType Combinations (sorted by median token count):")
        sorted_combinations = sorted(
            combination_stats.keys(), key=lambda x: combination_stats[x]["median"]
        )
        for combination in sorted_combinations:
            s = combination_stats[combination]
            small_count = sum(s["distribution"][r]["count"] for r in ["≤1K", "1K-2K"])
            size_info = f"(size {s['combination_size']})"
            print(
                f"  {combination:20} {size_info:9}: median {s['median']:>5,.0f} tokens, {small_count:>3}/{s['total_count']} (≤2K tokens)"
            )

    print("=" * 80)


def main() -> None:
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze diff token distribution by commit type and combinations"
    )
    parser.add_argument(
        "--csv-file",
        default="data/tangled_css_dataset.csv",
        help="Path to CSV file to analyze",
    )
    parser.add_argument(
        "--type-column", default="types", help="Column name containing commit types"
    )
    parser.add_argument(
        "--diff-column", default="diff", help="Column name containing diff content"
    )
    parser.add_argument(
        "--output",
        default="token_distribution_analysis.md",
        help="Output markdown file name",
    )

    args = parser.parse_args()

    csv_file = Path(args.csv_file)
    output_file = Path(args.output)

    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        return

    print(f"Git Diff Token Distribution Analyzer")
    print(f"Loading dataset from {csv_file}...")

    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records")
        print(f"Using type column: {args.type_column}")
        print(f"Using diff column: {args.diff_column}")
        print("Note: Using tiktoken cl100k_base encoding for token counting\n")

        # Create expanded dataset
        print("Creating expanded dataset...")
        expanded_df = create_expanded_dataset(df, args.type_column, args.diff_column)
        print(f"Expanded to {len(expanded_df)} type-diff pairs")

        # Analyze individual types
        individual_stats = analyze_individual_types(expanded_df)

        # Analyze type combinations
        combination_stats = analyze_type_combinations(expanded_df)

        # Generate report
        print(f"\nGenerating markdown report: {output_file}")
        generate_summary_report(individual_stats, combination_stats, output_file)

        # Print summary
        print_analysis_summary(individual_stats, combination_stats)

        print(f"\nAnalysis complete! Report saved to {output_file}")

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
