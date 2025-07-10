import csv
import logging
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


CONVENTIONAL_COMMIT_TYPES = [
    "feat",
    "fix",
    "refactor",
    "test",
    "docs",
    "build",
    "cicd",
]
SAMPLES_PER_TYPE = 30
OUTPUT_COLUMNS = ["annotated_type", "masked_commit_message", "git_diff", "sha"]


def load_sha_backup(file_path: Path) -> Set[str]:
    """Load SHA backup file and return set of SHAs to exclude."""
    if not file_path.exists():
        logging.warning(f"SHA backup file not found: {file_path}")
        return set()

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        sha_set = set(df["sha"].astype(str))
        logging.info(f"Loaded {len(sha_set)} SHAs to exclude from backup")
        return sha_set
    except Exception as e:
        logging.error(f"Error loading SHA backup: {e}")
        return set()


def load_ccs_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load CCS dataset CSV file using pandas for better handling of large fields."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        # Use pandas to read CSV which handles large fields better
        df = pd.read_csv(file_path, encoding="utf-8")
        data = df.to_dict("records")

        logging.info(f"Loaded {len(data)} records from CCS dataset")
        return data
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise


def save_to_csv(
    data: List[Dict[str, str]], output_path: Path, columns: List[str]
) -> None:
    """Save processed data to CSV file with specified columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)

    logging.info(f"Saved {len(data)} records to {output_path}")


def normalize_commit_type(commit_type: str) -> Optional[str]:
    """Normalize commit type according to mapping rules."""
    commit_type = commit_type.lower().strip()

    # Convert ci to cicd
    if commit_type == "ci":
        return "cicd"

    # Return type if it's in our valid types
    if commit_type in CONVENTIONAL_COMMIT_TYPES:
        return commit_type

    return None


def group_commits_by_type(
    dataset: List[Dict[str, Any]], valid_types: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Groups commits by their normalized concern type."""
    commits_by_type = {commit_type: [] for commit_type in valid_types}

    excluded_count = 0
    for item in dataset:
        commit_type = normalize_commit_type(item.get("annotated_type", ""))

        if commit_type is None:
            excluded_count += 1
            continue

        if commit_type in commits_by_type:
            commits_by_type[commit_type].append(item)

    logging.info(f"Excluded {excluded_count} records (style or invalid types)")

    for commit_type, commits in commits_by_type.items():
        logging.info(f"{commit_type}: {len(commits)} commits")

    return commits_by_type


def sample_commits_by_type(
    commits_by_type: Dict[str, List[Dict[str, Any]]],
    samples_per_type: int,
    output_columns: List[str],
) -> List[Dict[str, str]]:
    """Sample specified number of commits per type and extract required columns."""
    sampled_data = []

    for commit_type, commits in commits_by_type.items():
        if len(commits) == 0:
            logging.warning(f"No commits found for type: {commit_type}")
            continue

        # Sample with replacement if not enough commits
        if len(commits) < samples_per_type:
            logging.warning(
                f"Only {len(commits)} commits available for {commit_type}, "
                f"sampling with replacement"
            )
            sampled_commits = random.choices(commits, k=samples_per_type)
        else:
            sampled_commits = random.sample(commits, samples_per_type)

        # Extract only required columns
        for commit in sampled_commits:
            sampled_record = {}
            for column in output_columns:
                sampled_record[column] = commit.get(column, "")
            sampled_data.append(sampled_record)

    return sampled_data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Starting CCS dataset processing for concern extraction.")

    # Load SHA backup to exclude
    sha_backup_path = Path("ccs/sha_backup.csv")
    excluded_shas = load_sha_backup(sha_backup_path)

    # Load CCS dataset
    source_path = Path("ccs/CCS Dataset Training Data.csv")
    ccs_dataset = load_ccs_dataset(source_path)

    # Filter out commits with SHAs in backup
    original_count = len(ccs_dataset)
    ccs_dataset = [
        item for item in ccs_dataset if str(item.get("sha", "")) not in excluded_shas
    ]
    filtered_count = len(ccs_dataset)
    logging.info(
        f"Filtered out {original_count - filtered_count} commits from SHA backup"
    )

    # Group commits by normalized type
    commits_by_type = group_commits_by_type(ccs_dataset, CONVENTIONAL_COMMIT_TYPES)

    # Sample commits and extract required columns
    sampled_data = sample_commits_by_type(
        commits_by_type, SAMPLES_PER_TYPE, OUTPUT_COLUMNS
    )

    logging.info(f"Generated {len(sampled_data)} samples total")

    # Save to CSV in ccs folder
    output_path = Path("ccs/sampled_commits.csv")
    save_to_csv(sampled_data, output_path, OUTPUT_COLUMNS)

    # Print summary
    type_counts = {}
    for record in sampled_data:
        commit_type = record.get("annotated_type", "")
        type_counts[commit_type] = type_counts.get(commit_type, 0) + 1

    print("\nSample distribution:")
    for commit_type in sorted(type_counts.keys()):
        print(f"  {commit_type}: {type_counts[commit_type]} samples")


if __name__ == "__main__":
    main()
