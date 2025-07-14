#!/usr/bin/env python3
"""
Script to clean sampled_ccs_dataset.csv by removing:
1. Wrong commits based on SHA values in wrong_commites.csv
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
WRONG_COMMITS_FILE: str = "data/excluded_commits.csv"
SAMPLED_COMMITS_FILE: str = "data/sampled_ccs_dataset.csv"
SHA_BACKUP_FILE: str = "data/processed_shas.csv"


def load_wrong_commit_shas(wrong_commits_file_path: str) -> Set[str]:
    """Load SHA values from wrong commits CSV file."""
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
    """Remove wrong commits, overflowed types, and token-exceeding commits from sampled commits CSV file."""
    try:
        # Load the sampled commits data
        print(f"Loading sampled commits from {sampled_commits_file_path}...")
        df = pd.read_csv(sampled_commits_file_path)
        initial_count = len(df)
        print(f"Loaded {initial_count} sampled commits")

        # Step 1: Filter out wrong commits by SHA
        if wrong_commit_shas:
            df = df[~df["sha"].isin(wrong_commit_shas)]
            after_wrong_removal = len(df)
            wrong_removed = initial_count - after_wrong_removal
            print(f"Removed {wrong_removed} wrong commits")
        else:
            after_wrong_removal = initial_count
            print("No wrong commits to remove")

        # Step 2: Remove overflowed types to achieve target count
        np.random.seed(RANDOM_SEED)
        type_counts = df["annotated_type"].value_counts()
        print(f"Current type counts: {type_counts.to_dict()}")

        overflowed_shas_to_remove = set()
        for commit_type, current_count in type_counts.items():
            if current_count > TARGET_COUNT_PER_TYPE:
                remove_count = current_count - TARGET_COUNT_PER_TYPE
                type_commits = df[df["annotated_type"] == commit_type]
                commits_to_remove = type_commits.sample(n=remove_count)
                overflowed_shas_to_remove.update(commits_to_remove["sha"].astype(str))
                print(
                    f"Selected {remove_count} {commit_type} commits for overflow removal"
                )

        if overflowed_shas_to_remove:
            df = df[~df["sha"].astype(str).isin(overflowed_shas_to_remove)]
            after_overflow_removal = len(df)
            overflow_removed = after_wrong_removal - after_overflow_removal
            print(f"Removed {overflow_removed} overflowed commits")
        else:
            after_overflow_removal = after_wrong_removal
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
        final_type_counts = df["annotated_type"].value_counts()
        print(f"Final type counts: {final_type_counts.to_dict()}")

        # Save back to the same file
        df.to_csv(sampled_commits_file_path, index=False)
        print(f"Updated {sampled_commits_file_path} with cleaned data")

        # Also update SHA backup if exists
        if os.path.exists(SHA_BACKUP_FILE):
            all_removed_shas = wrong_commit_shas | overflowed_shas_to_remove
            if all_removed_shas or token_exceeded_indices:
                backup_df = pd.read_csv(SHA_BACKUP_FILE)
                backup_df = backup_df[
                    ~backup_df["sha"].astype(str).isin(all_removed_shas)
                ]
                backup_df.to_csv(SHA_BACKUP_FILE, index=False)
                print(f"Updated {SHA_BACKUP_FILE}")

    except FileNotFoundError:
        print(f"Error: File {sampled_commits_file_path} not found")
        sys.exit(1)
    except KeyError:
        print(f"Error: Required columns not found in {sampled_commits_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing sampled commits file: {e}")
        sys.exit(1)


def main() -> None:
    """Main function to orchestrate the complete dataset cleaning process."""
    print("Starting comprehensive dataset cleaning process...")
    print(f"Target count per type: {TARGET_COUNT_PER_TYPE}")
    print(f"Target token limit: {TARGET_TOKEN_LIMIT:,}")
    print("=" * 60)

    # Check if input files exist
    if not os.path.exists(SAMPLED_COMMITS_FILE):
        print(f"Error: {SAMPLED_COMMITS_FILE} not found in current directory")
        sys.exit(1)

    # Load wrong commit SHAs (optional file)
    wrong_commit_shas = set()
    if os.path.exists(WRONG_COMMITS_FILE):
        wrong_commit_shas = load_wrong_commit_shas(WRONG_COMMITS_FILE)
        print(f"Found {len(wrong_commit_shas)} unique wrong commit SHAs to remove")
    else:
        print(f"No {WRONG_COMMITS_FILE} found, skipping wrong commit removal")

    # Process all cleaning steps
    remove_wrong_commits_inplace(SAMPLED_COMMITS_FILE, wrong_commit_shas)

    print("=" * 60)
    print("Comprehensive dataset cleaning completed successfully!")


if __name__ == "__main__":
    main()
