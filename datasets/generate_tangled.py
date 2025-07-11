import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Set
import pandas as pd
import numpy as np
import tiktoken

# Configuration constants
DATASET_PATH = Path("ccs/sampled_css_dataset.csv")
CASES_PER_CONCERN_COUNT = 210
CONCERN_COUNTS = [1, 2, 3]
TOTAL_CASES = CASES_PER_CONCERN_COUNT * len(CONCERN_COUNTS)  # 334 * 3 = 1002
SEED = 42
OUTPUT_PATH = Path("ccs/css_tangled_dataset.csv")

# Token limit for diff size (16384 - 4096 = 12288)
MAX_DIFF_TOKENS = 12288

# Known concern types from CCS dataset
CONCERN_TYPES = ["feat", "fix", "refactor", "test", "docs", "build", "cicd"]

# CSV schema columns
OUTPUT_COLUMNS = ["description", "diff", "concern_count", "shas", "types"]

# Column mapping for preprocessing
COLUMN_MAPPING = {
    "masked_commit_message": "message",
    "git_diff": "diff",
    "sha": "sha",
    "annotated_type": "type",
}

# Required columns for tangled generation
REQUIRED_COLUMNS = ["message", "diff", "sha", "type"]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load CCS sampled dataset CSV file as raw DataFrame."""
    df = pd.read_csv(file_path, encoding="utf-8")
    logging.info(f"Loaded {len(df)} records from CCS dataset")

    # Log unique concern types to verify dataset has expected types before sampling
    # This helps catch data quality issues early if types don't match CONCERN_TYPES
    available_types = df["annotated_type"].unique().tolist()
    logging.info(f"Available concern types: {available_types}")

    # Display type distribution using pandas value_counts
    type_counts = df["annotated_type"].value_counts()
    for change_type, count in type_counts.items():
        logging.info(f"{change_type}: {count} changes")

    return df


def preprocess_dataset(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Preprocess raw dataset: rename columns, filter, and group by concern type."""
    df_processed = df.rename(columns=COLUMN_MAPPING)
    logging.info(
        f"Renamed columns: {list(COLUMN_MAPPING.keys())} -> {list(COLUMN_MAPPING.values())}"
    )

    df_processed = df_processed[REQUIRED_COLUMNS]
    logging.info(f"Selected required columns: {REQUIRED_COLUMNS}")

    # Convert to NumPy structured array - eliminates pandas overhead, reduces memory usage
    data_array = df_processed.to_records(index=False)

    grouped_by_type = {}
    for concern_type in CONCERN_TYPES:
        type_mask = data_array["type"] == concern_type
        type_data = data_array[type_mask]

        grouped_by_type[concern_type] = type_data
        logging.info(
            f"Preprocessed {concern_type}: {len(type_data)} records ready for sampling"
        )

    return grouped_by_type


def generate_cases_for_concern_count(
    grouped_data: Dict[str, np.ndarray],
    concern_count: int,
    num_cases: int,
) -> List[Dict[str, Any]]:
    """Generate cases for a specific concern count using preprocessed data."""
    cases = []
    seen_sha_combinations: Set[frozenset] = set()
    attempts = 0
    token_rejected = 0  # Track cases rejected due to token limit
    duplicate_rejected = 0  # Track cases rejected due to SHA duplication
    # tiktoken encoder for counting tokens in diff content
    encoding = tiktoken.get_encoding("cl100k_base")

    logging.info(
        f"Starting generation for {concern_count} concerns (target: {num_cases} cases)"
    )

    while len(cases) < num_cases:
        attempts += 1

        # Show progress every 100 attempts
        if attempts % 100 == 0:
            logging.info(
                f"  Progress: {len(cases)}/{num_cases} cases, {attempts} attempts, "
                f"{token_rejected} token-rejected, {duplicate_rejected} duplicate-rejected"
            )

        selected_types = random.sample(CONCERN_TYPES, concern_count)
        atomic_changes = []
        for selected_type in selected_types:
            type_data = grouped_data[selected_type]
            random_index = np.random.choice(len(type_data))
            sampled_record = type_data[random_index]

            record_dict = {
                "message": str(sampled_record["message"]),
                "diff": str(sampled_record["diff"]),
                "sha": str(sampled_record["sha"]),
                "type": str(sampled_record["type"]),
            }
            atomic_changes.append(record_dict)

        # Data processing: extract and combine data from sampled records
        messages = [change["message"] for change in atomic_changes]
        diffs = [change["diff"] for change in atomic_changes]
        shas = [change["sha"] for change in atomic_changes]
        types = [change["type"] for change in atomic_changes]

        description = " | ".join(messages)
        diff = "\n".join(diffs)

        # Check diff token count using tiktoken - skip if exceeds limit
        diff_tokens = len(encoding.encode(diff))
        if diff_tokens > MAX_DIFF_TOKENS:
            token_rejected += 1
            continue

        sha_combination = frozenset(shas)
        if sha_combination not in seen_sha_combinations:
            seen_sha_combinations.add(sha_combination)
            cases.append(
                {
                    "description": description,
                    "diff": diff,
                    "concern_count": concern_count,
                    "shas": json.dumps(shas),
                    "types": json.dumps(types),
                }
            )
        else:
            duplicate_rejected += 1

    logging.info(
        f"Completed {concern_count} concerns: {len(cases)} cases after {attempts} attempts "
        f"(token-rejected: {token_rejected}, duplicate-rejected: {duplicate_rejected})"
    )
    return cases


def generate_tangled_cases(grouped_data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Generate tangled change cases using preprocessed grouped data.

    Uses list comprehension instead of extend() for better performance and readability.
    List comprehension creates the final list in one pass, avoiding repeated list
    resizing and copying that can occur with extend().

    Args:
        grouped_data: Dictionary mapping concern types to their NumPy arrays

    Returns:
        List of generated tangled change cases
    """
    all_cases = [
        case
        for concern_count in CONCERN_COUNTS
        for case in generate_cases_for_concern_count(
            grouped_data, concern_count, CASES_PER_CONCERN_COUNT
        )
    ]

    logging.info(f"Total generated cases: {len(all_cases)}")
    return all_cases


def save_to_csv(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Save data to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame and save to CSV using pandas
    df = pd.DataFrame(data, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8")

    logging.info(f"Saved {len(data)} records to {output_path}")


def print_detailed_summary(cases: List[Dict[str, Any]]) -> None:
    """Print detailed summary of generated cases."""
    # Convert cases list to DataFrame for easier analysis using pandas
    df = pd.DataFrame(cases)

    print(f"\nGenerated {len(cases)} total tangled cases:")
    print(f"Target: {TOTAL_CASES} cases ({CASES_PER_CONCERN_COUNT} per concern count)")

    # Concern count distribution using pandas value_counts and sort_index
    concern_counts = df["concern_count"].value_counts().sort_index()
    print(f"\nConcern count distribution:")
    for concern_count, count in concern_counts.items():
        percentage = (count / len(cases)) * 100
        print(f"  {concern_count} concerns: {count} cases ({percentage:.1f}%)")

    # Type distribution analysis using NumPy for better performance
    # NumPy array creation - faster than list operations for large datasets
    all_types = []
    for case_types_str in df["types"]:
        case_types = json.loads(case_types_str)
        all_types.extend(case_types)

    unique_types, type_counts = np.unique(all_types, return_counts=True)

    print(f"\nType distribution (total atomic changes):")
    sorted_indices = np.argsort(unique_types)
    for idx in sorted_indices:
        change_type = unique_types[idx]
        count = type_counts[idx]
        print(f"  {change_type}: {count} atomic changes")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Starting complete CSS tangled dataset generation.")
    logging.info(f"Target: {CASES_PER_CONCERN_COUNT} cases per concern count (1, 2, 3)")
    logging.info(f"Total target cases: {TOTAL_CASES}")
    logging.info(f"Known concern types: {CONCERN_TYPES}")

    # Set global random seeds for complete reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # Step 1: Load raw dataset from CSV
    df = load_dataset(DATASET_PATH)

    # Step 2: Preprocess dataset (rename, filter, group by type)
    grouped_data = preprocess_dataset(df)

    # Step 3: Generate tangled cases using preprocessed data
    cases = generate_tangled_cases(grouped_data)

    # Step 4: Save results to CSV
    save_to_csv(cases, OUTPUT_PATH)

    # Step 5: Print detailed summary
    print_detailed_summary(cases)


if __name__ == "__main__":
    main()
