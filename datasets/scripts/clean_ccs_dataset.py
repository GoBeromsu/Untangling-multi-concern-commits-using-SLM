#!/usr/bin/env python3
"""
Script to clean sampled_ccs_dataset.csv by removing:
1. excluded commits based on SHA values in excluded_commites.csv
2. Overflowed types to achieve target count
3. Commits exceeding token limit
"""

import pandas as pd
import numpy as np
import tiktoken
from typing import Set
import sys
import os

# Processing configuration
TARGET_COUNT_PER_TYPE: int = 50
TARGET_TOKEN_LIMIT: int = 12288  # 16384 - 4096
ENCODING_MODEL: str = "cl100k_base"  # GPT-4 encoding
RANDOM_SEED: int = 42

# File paths
EXCLUDED_COMMITS_FILE: str = "data/excluded_commits.csv"
SAMPLED_COMMITS_FILE: str = "data/sampled_ccs_dataset.csv"
SHA_BACKUP_FILE: str = "data/processed_shas.csv"

# DataFrame column names
SHA_COLUMN: str = "sha"
ANNOTATED_TYPE_COLUMN: str = "annotated_type"


def load_excluded_commit_shas(excluded_commits_file_path: str) -> Set[str]:
    """Load SHA values from excluded commits CSV file."""
    excluded_commits_df = pd.read_csv(excluded_commits_file_path)
    print(
        f"Loaded {len(excluded_commits_df)} excluded commits from {excluded_commits_file_path}"
    )
    return set(excluded_commits_df[SHA_COLUMN].tolist())


def remove_excluded_commits(
    sampled_commits_file_path: str, excluded_commit_shas: Set[str]
) -> None:
    """Remove excluded commits, overflowed types, and token-exceeding commits from sampled commits CSV file."""
    # Load the sampled commits data
    print(f"Loading sampled commits from {sampled_commits_file_path}...")
    df = pd.read_csv(sampled_commits_file_path)
    initial_count = len(df)
    print(f"Loaded {initial_count} sampled commits")

    # Step 1: Filter out excluded commits by SHA
    # Remove commits that have SHA values matching those in the excluded list
    df = df[~df[SHA_COLUMN].isin(excluded_commit_shas)]
    after_excluded_removal = len(df)
    excluded_removed = initial_count - after_excluded_removal
    print(f"Removed {excluded_removed} excluded commits")

    # Step 2: Remove overflowed types to achieve target count
    np.random.seed(RANDOM_SEED)
    type_counts = df[ANNOTATED_TYPE_COLUMN].value_counts()
    print(f"Current type counts: {type_counts.to_dict()}")

    overflowed_shas_to_remove = set()
    for commit_type, current_count in type_counts.items():
        if current_count > TARGET_COUNT_PER_TYPE:
            remove_count = current_count - TARGET_COUNT_PER_TYPE
            type_commits = df[df[ANNOTATED_TYPE_COLUMN] == commit_type]
            commits_to_remove = type_commits.sample(n=remove_count)
            overflowed_shas_to_remove.update(commits_to_remove[SHA_COLUMN].astype(str))
            print(f"Selected {remove_count} {commit_type} commits for overflow removal")

    if overflowed_shas_to_remove:
        df = df[~df[SHA_COLUMN].astype(str).isin(overflowed_shas_to_remove)]
        after_overflow_removal = len(df)
        overflow_removed = after_excluded_removal - after_overflow_removal
        print(f"Removed {overflow_removed} overflowed commits")
    else:
        after_overflow_removal = after_excluded_removal
        print("No overflowed commits to remove")

    # Step 3: Remove commits exceeding token limit
    encoding = tiktoken.get_encoding(ENCODING_MODEL)
    token_exceeded_indices = []

    for idx, row in df.iterrows():
        combined_text = " ".join(
            [str(row[col]) if pd.notna(row[col]) else "" for col in df.columns]
        )
        token_count = len(encoding.encode(combined_text))

        if token_count > TARGET_TOKEN_LIMIT:
            token_exceeded_indices.append(idx)

    if token_exceeded_indices:
        df = df.drop(token_exceeded_indices)
        final_count = len(df)
        token_removed = after_overflow_removal - final_count
        print(
            f"Removed {token_removed} commits exceeding {TARGET_TOKEN_LIMIT:,} tokens"
        )
    else:
        final_count = after_overflow_removal
        print(f"No commits exceed {TARGET_TOKEN_LIMIT:,} token limit")

    # Summary
    total_removed = initial_count - final_count
    print(f"Total removed: {total_removed} commits")
    print(f"Remaining: {final_count} valid commits")

    # Show final type counts
    final_type_counts = df[ANNOTATED_TYPE_COLUMN].value_counts()
    print(f"Final type counts: {final_type_counts.to_dict()}")

    # Save back to the same file
    df.to_csv(sampled_commits_file_path, index=False)
    print(f"Updated {sampled_commits_file_path} with cleaned data")

    # Also update SHA backup if exists
    if os.path.exists(SHA_BACKUP_FILE):
        all_removed_shas = excluded_commit_shas | overflowed_shas_to_remove
        if all_removed_shas or token_exceeded_indices:
            backup_df = pd.read_csv(SHA_BACKUP_FILE)
            backup_df = backup_df[
                ~backup_df[SHA_COLUMN].astype(str).isin(all_removed_shas)
            ]
            backup_df.to_csv(SHA_BACKUP_FILE, index=False)
            print(f"Updated {SHA_BACKUP_FILE}")


def main() -> None:
    """Main function to orchestrate the complete dataset cleaning process."""
    print("Starting comprehensive dataset cleaning process...")
    print(f"Target count per type: {TARGET_COUNT_PER_TYPE}")
    print(f"Target token limit: {TARGET_TOKEN_LIMIT:,}")
    print("=" * 60)

    try:
        # Check if required input file exists
        if not os.path.exists(SAMPLED_COMMITS_FILE):
            raise FileNotFoundError(
                f"{SAMPLED_COMMITS_FILE} not found in current directory"
            )

        # Load excluded commit SHAs (optional file)
        excluded_commit_shas = set()
        if os.path.exists(EXCLUDED_COMMITS_FILE):
            excluded_commit_shas = load_excluded_commit_shas(EXCLUDED_COMMITS_FILE)
            print(
                f"Found {len(excluded_commit_shas)} unique excluded commit SHAs to remove"
            )
        else:
            print(f"No {EXCLUDED_COMMITS_FILE} found, skipping excluded commit removal")

        # Process all cleaning steps
        remove_excluded_commits(SAMPLED_COMMITS_FILE, excluded_commit_shas)

        print("=" * 60)
        print("Comprehensive dataset cleaning completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Required column not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during dataset cleaning: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
