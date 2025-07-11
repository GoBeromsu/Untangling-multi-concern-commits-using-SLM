#!/usr/bin/env python3
"""
Script to remove wrong commits from sampled_css_dataset.csv based on SHA values in wrong_commites.csv
"""

import pandas as pd
from typing import Set
import sys
import os


def load_wrong_commit_shas(wrong_commits_file_path: str) -> Set[str]:
    """
    Load SHA values from wrong commits CSV file.

    Args:
        wrong_commits_file_path: Path to the wrong commits CSV file

    Returns:
        Set of SHA values to be removed
    """
    try:
        wrong_commits_df = pd.read_csv(wrong_commits_file_path)
        print(
            f"Loaded {len(wrong_commits_df)} wrong commits from {wrong_commits_file_path}"
        )
        return set(wrong_commits_df["sha"].tolist())
    except FileNotFoundError:
        print(f"Error: File {wrong_commits_file_path} not found")
        sys.exit(1)
    except KeyError:
        print(f"Error: 'sha' column not found in {wrong_commits_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading wrong commits file: {e}")
        sys.exit(1)


def remove_wrong_commits_inplace(
    sampled_commits_file_path: str, wrong_commit_shas: Set[str]
) -> None:
    """
    Remove wrong commits from sampled commits CSV file and save back to the same file.

    Args:
        sampled_commits_file_path: Path to the sampled commits CSV file
        wrong_commit_shas: Set of SHA values to remove
    """
    try:
        # Load the sampled commits data
        print(f"Loading sampled commits from {sampled_commits_file_path}...")
        sampled_commits_df = pd.read_csv(sampled_commits_file_path)
        print(f"Loaded {len(sampled_commits_df)} sampled commits")

        # Filter out wrong commits by SHA
        initial_count = len(sampled_commits_df)
        filtered_df = sampled_commits_df[
            ~sampled_commits_df["sha"].isin(wrong_commit_shas)
        ]
        final_count = len(filtered_df)
        removed_count = initial_count - final_count

        print(f"Removed {removed_count} wrong commits")
        print(f"Remaining {final_count} valid commits")

        # Save back to the same file
        filtered_df.to_csv(sampled_commits_file_path, index=False)
        print(f"Updated {sampled_commits_file_path} with cleaned data")

    except FileNotFoundError:
        print(f"Error: File {sampled_commits_file_path} not found")
        sys.exit(1)
    except KeyError:
        print(f"Error: 'sha' column not found in {sampled_commits_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing sampled commits file: {e}")
        sys.exit(1)


def main() -> None:
    """Main function to orchestrate the wrong commit removal process."""

    # File paths
    wrong_commits_file = "wrong_commites.csv"
    sampled_commits_file = "sampled_css_dataset.csv"

    # Check if input files exist
    if not os.path.exists(wrong_commits_file):
        print(f"Error: {wrong_commits_file} not found in current directory")
        sys.exit(1)

    if not os.path.exists(sampled_commits_file):
        print(f"Error: {sampled_commits_file} not found in current directory")
        sys.exit(1)

    print("Starting wrong commit removal process...")
    print("=" * 50)

    # Load wrong commit SHAs
    wrong_commit_shas = load_wrong_commit_shas(wrong_commits_file)
    print(f"Found {len(wrong_commit_shas)} unique wrong commit SHAs to remove")

    # Remove wrong commits from sampled data in-place
    remove_wrong_commits_inplace(sampled_commits_file, wrong_commit_shas)

    print("=" * 50)
    print("Wrong commit removal process completed successfully!")


if __name__ == "__main__":
    main()
