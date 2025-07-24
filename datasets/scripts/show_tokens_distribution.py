#!/usr/bin/env python3
"""
Analyze git diff token distribution for tangled commits.
Purpose: Find token size distribution to optimize LLM input.
"""

import pandas as pd
import numpy as np
import tiktoken
from pathlib import Path

# Constants
ENCODING_NAME = "cl100k_base"
CSV_FILE = "../data/tangled_ccs_dataset_test.csv"
DIFF_COLUMN = "diff"
OUTPUT_FILE = "../token_distribution_results.csv"


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(ENCODING_NAME)
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def get_token_range(token_count: int) -> str:
    """Get token range label."""
    if token_count <= 1024:
        return "≤1024"
    elif token_count <= 2048:
        return "1025-2048"
    elif token_count <= 4096:
        return "2049-4096"
    elif token_count <= 8192:
        return "4097-8192"
    elif token_count <= 16384:
        return "8193-16384"
    elif token_count <= 32768:
        return "16385-32768"
    else:
        return ">32768"


def analyze_token_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze token distribution of tangled commits."""
    results = []

    for idx, row in df.iterrows():
        diff_text = row[DIFF_COLUMN]

        if pd.isna(diff_text):
            continue

        token_count = count_tokens(str(diff_text))
        token_range = get_token_range(token_count)

        results.append(
            {
                "row_idx": idx,
                "token_count": token_count,
                "token_range": token_range,
            }
        )

    return pd.DataFrame(results)


def create_distribution_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create token distribution summary."""
    token_ranges = [
        "≤1024",
        "1025-2048",
        "2049-4096",
        "4097-8192",
        "8193-16384",
        "16385-32768",
        ">32768",
    ]

    total_count = len(results_df)
    distribution = results_df["token_range"].value_counts()
    token_counts = results_df["token_count"].values

    summary_data = []
    for range_label in token_ranges:
        count = distribution.get(range_label, 0)
        percentage = (count / total_count) * 100
        summary_data.append(
            {
                "token_range": range_label,
                "count": count,
                "percentage": round(percentage, 1),
            }
        )

    # Add overall statistics
    stats = {
        "total_samples": total_count,
        "median_tokens": int(np.median(token_counts)),
        "mean_tokens": int(np.mean(token_counts)),
        "min_tokens": int(np.min(token_counts)),
        "max_tokens": int(np.max(token_counts)),
    }

    return pd.DataFrame(summary_data), stats


def main() -> None:
    """Main analysis function."""
    csv_file = Path(CSV_FILE)
    output_file = Path(OUTPUT_FILE)

    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        return

    print(f"Tangled Commits Token Distribution Analyzer")
    print(f"Loading dataset from {csv_file}...")

    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} tangled commits")
        print(f"Using tiktoken {ENCODING_NAME} encoding")

        print("Analyzing token distribution...")
        results_df = analyze_token_distribution(df)
        print(f"Processed {len(results_df)} diffs")

        print("Creating distribution summary...")
        summary_df, stats = create_distribution_summary(results_df)

        # Save detailed results
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

        # Print summary
        print("\nToken Distribution Summary:")
        print("=" * 60)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Median tokens: {stats['median_tokens']}")
        print(f"Mean tokens: {stats['mean_tokens']}")
        print(f"Min tokens: {stats['min_tokens']}")
        print(f"Max tokens: {stats['max_tokens']}")
        print("-" * 60)

        for _, row in summary_df.iterrows():
            print(
                f"{row['token_range']:>12}: {row['count']:>4} samples ({row['percentage']:>5.1f}%)"
            )

        # Key thresholds
        le_1024 = summary_df[summary_df["token_range"] == "≤1024"]["percentage"].iloc[0]
        le_4096 = summary_df[
            summary_df["token_range"].isin(["≤1024", "1025-2048", "2049-4096"])
        ]["percentage"].sum()
        le_8192 = summary_df[
            summary_df["token_range"].isin(
                ["≤1024", "1025-2048", "2049-4096", "4097-8192"]
            )
        ]["percentage"].sum()

        print("-" * 60)
        print(f"≤1024 tokens: {le_1024:.1f}%")
        print(f"≤4096 tokens: {le_4096:.1f}%")
        print(f"≤8192 tokens: {le_8192:.1f}%")

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
