#!/usr/bin/env python3
"""Extract git diff files from CCS sampled commits CSV by type."""


import pandas as pd
from pathlib import Path
from typing import Dict, List


def create_type_directories(base_output_dir: Path, types: List[str]) -> Dict[str, Path]:
    """Create directories for each type."""
    type_dirs = {}
    for type_name in types:
        type_dir = base_output_dir / type_name
        type_dir.mkdir(parents=True, exist_ok=True)
        type_dirs[type_name] = type_dir
    return type_dirs


def save_diff_files(df: pd.DataFrame, type_dirs: Dict[str, Path]) -> Dict[str, int]:
    """Save git diffs organized by type into separate directories."""
    type_counts = {}

    for _, row in df.iterrows():
        commit_type = row["annotated_type"]
        commit_message = row["masked_commit_message"]
        git_diff = row["git_diff"]
        sha = row["sha"]

        if commit_type not in type_dirs:
            continue

        # Count entries for this type
        if commit_type not in type_counts:
            type_counts[commit_type] = 0

        type_counts[commit_type] += 1
        entry_num = type_counts[commit_type]

        # Generate filename using type + sha for uniqueness
        filename = f"{commit_type}_{entry_num}_{sha}.diff"
        filepath = type_dirs[commit_type] / filename

        # Create file content with metadata
        content_lines = [
            f"# Type: {commit_type}",
            f"# Commit Message: {commit_message}",
            f"# SHA: {sha}",
            "",
            "# === Git Diff Content ===",
            "",
            git_diff,
        ]

        try:
            with open(filepath, "w", encoding="utf-8") as file:
                file.write("\n".join(content_lines))
        except Exception as e:
            print(f"Error saving {filepath}: {e}")

    return type_counts


def main():
    """Main function to extract all git diffs by type from CCS CSV."""
    print("CCS Git Diff Extractor - Processing sampled commits...")

    # Configuration
    csv_file = Path("sampled_css_dataset.csv")
    base_output_dir = Path("extracted_diffs_by_type")

    if not csv_file.exists():
        print(f"Error: {csv_file} not found!")
        return

    # Load CSV data
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} commits from CSV")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Get unique types
    unique_types = sorted(df["annotated_type"].unique())
    print(f"Types found: {unique_types}")

    # Create output directories
    type_dirs = create_type_directories(base_output_dir, unique_types)

    # Save diff files
    type_counts = save_diff_files(df, type_dirs)

    # Display summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total commits processed: {len(df)}")
    print("\nBreakdown by type:")

    for commit_type, count in sorted(type_counts.items()):
        print(f"  {commit_type:12}: {count:3d} files")

    print("=" * 60)
    print(f"\nAll files saved to: {base_output_dir}")
    print("Process completed! 🎉")


if __name__ == "__main__":
    main()
