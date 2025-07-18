#!/usr/bin/env python3
"""
Script to sample atomic commits from CCS dataset for concern extraction.
Applies atomic sampling strategy with token filtering and SHA deduplication.
"""

import pandas as pd
import tiktoken
from typing import Dict, List, Set

# Processing configuration
CONVENTIONAL_COMMIT_TYPES: List[str] = ["cicd", "refactor", "fix", "test"]
SAMPLES_PER_TYPE: int = 2
TARGET_TOKEN_LIMIT: int = 12288  # 16384 - 4096
ENCODING_MODEL: str = "cl100k_base"  # GPT-4 encoding

# Column name constants
COLUMN_SHA: str = "sha"
COLUMN_ANNOTATED_TYPE: str = "annotated_type"
COLUMN_GIT_DIFF: str = "git_diff"
COLUMN_MASKED_COMMIT_MESSAGE: str = "masked_commit_message"
OUTPUT_COLUMNS: List[str] = [
    COLUMN_ANNOTATED_TYPE,
    COLUMN_MASKED_COMMIT_MESSAGE,
    COLUMN_GIT_DIFF,
    COLUMN_SHA,
]

# Data transformation constants
CI_TO_CICD_REPLACEMENT: str = "cicd"

# File paths
CCS_SOURCE_PATH: str = "data/CCS Dataset Training Data.csv"
SAMPLED_CSV_PATH: str = "data/sampled_ccs_dataset.csv"
DIFF_OUTPUT_DIR: str = "data/types"


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply CI to CICD label normalization using pandas vectorized operations."""
    # Use pandas replace for vectorized string replacement
    df[COLUMN_ANNOTATED_TYPE] = (
        df[COLUMN_ANNOTATED_TYPE]
        .str.lower()
        .str.strip()
        .replace("ci", CI_TO_CICD_REPLACEMENT)
    )
    print("Applied CI -> CICD normalization using pandas replace()")
    return df


def apply_token_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply token-based filtering using GPT-4 tokenizer with pandas operations."""
    encoding = tiktoken.get_encoding(ENCODING_MODEL)

    # Create combined text column for token counting using pandas string operations
    combined_text = (
        df[COLUMN_GIT_DIFF].astype(str)
        + " "
        + df[COLUMN_MASKED_COMMIT_MESSAGE].astype(str)
    )

    # Apply token counting function and create boolean mask using pandas apply()
    token_counts = combined_text.apply(lambda x: len(encoding.encode(x)))
    token_mask = token_counts <= TARGET_TOKEN_LIMIT

    # Filter using pandas boolean indexing
    filtered_df = df[token_mask].copy()

    removed_count = len(df) - len(filtered_df)
    if removed_count > 0:
        print(
            f"Token filtering: removed {removed_count} commits exceeding {TARGET_TOKEN_LIMIT} tokens using pandas boolean indexing"
        )

    print(f"Token filtering: kept {len(filtered_df)} commits")
    return filtered_df


def apply_sha_deduplication(df: pd.DataFrame, excluded_shas: Set[str]) -> pd.DataFrame:
    """Apply SHA deduplication using pandas isin() for efficient filtering."""
    original_count = len(df)

    # Use pandas isin() for vectorized membership testing
    sha_mask = ~df[COLUMN_SHA].astype(str).isin(excluded_shas)
    filtered_df = df[sha_mask].copy()

    removed_count = original_count - len(filtered_df)
    print(
        f"SHA deduplication: removed {removed_count} duplicate commits using pandas isin()"
    )
    return filtered_df


def load_existing_shas(file_path: str) -> Set[str]:
    """Load existing SHAs from sampled dataset to exclude duplicates."""
    try:
        df = pd.read_csv(file_path)
        sha_set = set(df[COLUMN_SHA].astype(str))
        print(f"Loaded {len(sha_set)} SHAs for deduplication")
        return sha_set
    except FileNotFoundError:
        print(f"No existing samples found at {file_path}")
        return set()
    except Exception as e:
        print(f"Error loading existing SHAs: {e}")
        return set()


def load_ccs_dataset(file_path: str) -> pd.DataFrame:
    """Load CCS dataset CSV file as pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Dataset is empty")

        required_columns = set(OUTPUT_COLUMNS)
        available_columns = set(df.columns)

        missing_columns = required_columns - available_columns
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print(f"Dataset validation passed: {len(df)} records with required columns")

        print(f"Loaded {len(df)} records from CCS dataset as DataFrame")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def save_to_csv(
    data: List[Dict[str, str]], output_path: str, columns: List[str]
) -> None:
    """Save processed data to CSV file."""
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if data:
        df = pd.DataFrame(data, columns=columns)
        file_exists = os.path.exists(output_path)

        df.to_csv(
            output_path,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
        )

    print(f"Saved {len(data)} records to {output_path}")


def group_commits_by_type(
    df: pd.DataFrame, valid_types: List[str]
) -> Dict[str, pd.DataFrame]:
    """Group commits by their concern type using pandas groupby."""
    # Filter valid types using pandas isin() for vectorized filtering
    type_mask = df[COLUMN_ANNOTATED_TYPE].isin(valid_types)
    valid_df = df[type_mask].copy()

    excluded_count = len(df) - len(valid_df)
    print(
        f"Type filtering: excluded {excluded_count} records (invalid types) using pandas isin()"
    )

    # Use pandas groupby for efficient grouping
    commits_by_type = {}
    for commit_type, group_df in valid_df.groupby(COLUMN_ANNOTATED_TYPE):
        commits_by_type[commit_type] = group_df
        print(f"  {commit_type}: {len(group_df)} commits")

    return commits_by_type


def sample_commits_for_type(
    df: pd.DataFrame, count: int, output_columns: List[str]
) -> List[Dict[str, str]]:
    """Randomly sample specified number of commits using pandas sample()."""
    # Use pandas sample() for efficient random sampling
    sampled_df = df.sample(n=count, random_state=None)

    # Convert only the final result to dict list for compatibility
    sampled_data = sampled_df[output_columns].to_dict("records")
    return sampled_data


def extract_diffs(sampled_data: List[Dict[str, str]], output_dir: str) -> None:
    """Extract git diff files organized by type into separate directories."""
    import os

    type_counts = {}

    for record in sampled_data:
        commit_type = record[COLUMN_ANNOTATED_TYPE]

        # Create type directory if needed
        type_dir = os.path.join(output_dir, commit_type)
        os.makedirs(type_dir, exist_ok=True)

        # Count entries for this type
        if commit_type not in type_counts:
            type_counts[commit_type] = 0
        type_counts[commit_type] += 1

        # Generate filename
        filename = f"{commit_type}_{type_counts[commit_type]}_{record[COLUMN_SHA]}.diff"
        filepath = os.path.join(type_dir, filename)

        # Create file content with metadata
        content_lines = [
            f"# Type: {commit_type}",
            f"# Commit Message: {record[COLUMN_MASKED_COMMIT_MESSAGE]}",
            f"# SHA: {record[COLUMN_SHA]}",
            "",
            "# === Git Diff Content ===",
            "",
            record[COLUMN_GIT_DIFF],
        ]

        with open(filepath, "w") as f:
            f.write("\n".join(content_lines))

    print(f"Extracted {len(sampled_data)} diff files to {output_dir}")


def main() -> None:
    """
    Main function implementing atomic sampling strategy:
    1. Load dataset and backup SHAs
    2. Apply CI->CICD normalization
    3. Apply token-based filtering
    4. Apply SHA deduplication
    5. Group by type and randomly sample
    6. Save results and extract diffs
    """
    print("Starting atomic sampling strategy for CCS dataset")
    print("=" * 50)

    # Step 1: Load dataset and backup SHAs
    print("Step 1: Loading dataset and backup SHAs")
    excluded_shas = load_existing_shas(SAMPLED_CSV_PATH)
    ccs_df = load_ccs_dataset(CCS_SOURCE_PATH)

    # Step 2: Apply CI->CICD normalization
    print("\nStep 2: Applying CI->CICD normalization")
    ccs_df = normalize_dataset(ccs_df)

    # Step 3: Apply token-based filtering
    print("\nStep 3: Applying token-based filtering")
    ccs_df = apply_token_filtering(ccs_df)

    # Step 4: Apply SHA deduplication
    print("\nStep 4: Applying SHA deduplication")
    ccs_df = apply_sha_deduplication(ccs_df, excluded_shas)

    # Step 5: Group by type and randomly sample
    print("\nStep 5: Grouping by type and random sampling")
    commits_by_type = group_commits_by_type(ccs_df, CONVENTIONAL_COMMIT_TYPES)

    all_sampled_data = []
    for commits_df in commits_by_type.values():
        sampled_data = sample_commits_for_type(
            commits_df, SAMPLES_PER_TYPE, OUTPUT_COLUMNS
        )
        all_sampled_data.extend(sampled_data)

    print(f"Random sampling: generated {len(all_sampled_data)} samples total")

    # Step 6: Save results and extract diffs
    print("\nStep 6: Saving results and extracting diffs")
    save_to_csv(all_sampled_data, SAMPLED_CSV_PATH, OUTPUT_COLUMNS)
    extract_diffs(all_sampled_data, DIFF_OUTPUT_DIR)

    # Final summary
    print("\n" + "=" * 50)
    print("Atomic sampling completed successfully!")

    type_counts = {}
    for record in all_sampled_data:
        commit_type = record.get(COLUMN_ANNOTATED_TYPE, "")
        type_counts[commit_type] = type_counts.get(commit_type, 0) + 1

    print("Final sample distribution:")
    for commit_type in sorted(type_counts.keys()):
        print(f"  {commit_type}: {type_counts[commit_type]} samples")


if __name__ == "__main__":
    main()
