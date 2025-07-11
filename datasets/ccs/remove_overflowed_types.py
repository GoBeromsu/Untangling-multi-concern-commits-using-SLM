#!/usr/bin/env python3
"""
Script to randomly remove commits of specified type to achieve target count of 30.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Set


# Constants
TARGET_COUNT = 30
RANDOM_SEED = 42

# Path constants
SAMPLED_COMMITS_PATH = Path("sampled_css_dataset.csv")
SHA_BACKUP_PATH = Path("sha_backup.csv")


def get_random_commit_shas_to_remove(
    sampled_commits_path: Path, commit_type: str
) -> Set[str]:
    """Get random commit SHAs of specified type to remove to reach target count."""
    if not sampled_commits_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {sampled_commits_path}")

    try:
        df = pd.read_csv(sampled_commits_path, encoding="utf-8")
        type_commits_df = df[df["annotated_type"] == commit_type]
        current_count = len(type_commits_df)

        logging.info(f"Found {current_count} {commit_type} commits")

        if current_count <= TARGET_COUNT:
            logging.info(
                f"Already at or below target ({TARGET_COUNT}). No removal needed."
            )
            return set()

        remove_count = current_count - TARGET_COUNT
        logging.info(
            f"Need to remove {remove_count} {commit_type} commits to reach target of {TARGET_COUNT}"
        )

        # Randomly sample commits to remove
        np.random.seed(RANDOM_SEED)
        commits_to_remove = type_commits_df.sample(n=remove_count)

        logging.info(f"Selected {remove_count} random {commit_type} commits to remove")
        return set(commits_to_remove["sha"].astype(str))

    except Exception as e:
        logging.error(f"Error loading sampled commits: {e}")
        raise


def remove_commits_by_sha(file_path: Path, shas_to_remove: Set[str]) -> None:
    """Remove specified commits from CSV file and save back to the same file."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        initial_count = len(df)

        # Filter out commits to remove
        filtered_df = df[~df["sha"].astype(str).isin(shas_to_remove)]
        final_count = len(filtered_df)
        removed_count = initial_count - final_count

        logging.info(f"Removed {removed_count} commits from {file_path}")
        logging.info(f"Remaining {final_count} entries")

        # For sampled_css_dataset.csv, show type count after removal
        if "annotated_type" in filtered_df.columns:
            type_counts = filtered_df["annotated_type"].value_counts()
            logging.info(f"Remaining commit counts by type: {type_counts.to_dict()}")

        # Save back to the same file
        filtered_df.to_csv(file_path, index=False, encoding="utf-8")
        logging.info(f"Updated {file_path}")

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        raise


def main() -> None:
    """Main function to orchestrate the random commit removal process."""
    parser = argparse.ArgumentParser(
        description="Remove random commits of specified type to reach target count"
    )
    parser.add_argument(
        "commit_type", help="Type of commits to process (e.g., feat, fix, docs)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(f"Starting random {args.commit_type} commit removal process...")
    logging.info(f"Target: Reduce {args.commit_type} commits to {TARGET_COUNT}")

    # Check if input files exist
    if not SAMPLED_COMMITS_PATH.exists():
        raise FileNotFoundError(
            f"{SAMPLED_COMMITS_PATH} not found in current directory"
        )

    if not SHA_BACKUP_PATH.exists():
        raise FileNotFoundError(f"{SHA_BACKUP_PATH} not found in current directory")

    # Get random commits to remove
    shas_to_remove = get_random_commit_shas_to_remove(
        SAMPLED_COMMITS_PATH, args.commit_type
    )

    if not shas_to_remove:
        logging.info("No commits to remove. Process completed.")
        return

    logging.info(f"SHAs to remove: {len(shas_to_remove)} commits")

    # Remove from both files
    remove_commits_by_sha(SAMPLED_COMMITS_PATH, shas_to_remove)
    remove_commits_by_sha(SHA_BACKUP_PATH, shas_to_remove)

    logging.info("Random commit removal process completed successfully!")


if __name__ == "__main__":
    main()
