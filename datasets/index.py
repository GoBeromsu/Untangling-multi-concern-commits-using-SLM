import csv
import logging
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


CONVENTIONAL_COMMIT_TYPES = [
    "feat",
]
SAMPLES_PER_TYPE = 15
OUTPUT_COLUMNS = ["annotated_type", "masked_commit_message", "git_diff", "sha"]

# Path constants
CCS_SOURCE_PATH = Path("ccs/CCS Dataset Training Data.csv")
SHA_BACKUP_PATH = Path("ccs/sha_backup.csv")
SAMPLED_CSV_PATH = Path("ccs/sampled_commits.csv")
DIFF_OUTPUT_DIR = Path("ccs/extracted_diffs_by_type")


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
        df = pd.read_csv(file_path, encoding="utf-8")

        df["annotated_type"] = (
            df["annotated_type"].str.lower().str.strip().replace("ci", "cicd")
        )
        data = df.to_dict("records")

        logging.info(f"Loaded {len(data)} records from CCS dataset")
        return data
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise


def save_to_csv(
    data: List[Dict[str, str]], output_path: Path, columns: List[str]
) -> None:
    """Append processed data to CSV file with specified columns using pandas."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if data:
        df = pd.DataFrame(data, columns=columns)

        # Append if file exists, otherwise create new
        df.to_csv(
            output_path,
            mode="a" if output_path.exists() else "w",
            header=not output_path.exists(),
            index=False,
            encoding="utf-8",
        )

    logging.info(f"Saved {len(data)} records to {output_path}")


def group_commits_by_type(
    dataset: List[Dict[str, Any]], valid_types: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Groups commits by their concern type."""
    commits_by_type = {commit_type: [] for commit_type in valid_types}

    excluded_count = 0
    for item in dataset:
        commit_type = item.get("annotated_type", "").lower().strip()

        if commit_type not in valid_types:
            excluded_count += 1
            continue

        commits_by_type[commit_type].append(item)

    logging.info(f"Excluded {excluded_count} records (style or invalid types)")

    for commit_type, commits in commits_by_type.items():
        logging.info(f"{commit_type}: {len(commits)} commits")

    return commits_by_type


def sample_commits_for_type(
    commits: List[Dict[str, Any]],
    count: int,
    output_columns: List[str],
) -> List[Dict[str, str]]:
    """Sample specified number of commits for a single type and extract required columns."""
    sampled_commits = random.sample(commits, count)
    sampled_data = []

    for commit in sampled_commits:
        sampled_record = {}
        for column in output_columns:
            sampled_record[column] = commit.get(column, "")
        sampled_data.append(sampled_record)

    return sampled_data


def extract_diffs(sampled_data: List[Dict[str, str]], output_dir: Path) -> None:
    """Extract git diff files organized by type into separate directories."""
    type_counts = {}

    for record in sampled_data:
        commit_type = record["annotated_type"]

        # Create type directory if needed
        type_dir = output_dir / commit_type
        type_dir.mkdir(parents=True, exist_ok=True)

        # Count entries for this type
        if commit_type not in type_counts:
            type_counts[commit_type] = 0
        type_counts[commit_type] += 1

        # Generate filename
        filename = f"{commit_type}_{type_counts[commit_type]}_{record['sha']}.diff"
        filepath = type_dir / filename

        # Create file content with metadata
        content_lines = [
            f"# Type: {commit_type}",
            f"# Commit Message: {record['masked_commit_message']}",
            f"# SHA: {record['sha']}",
            "",
            "# === Git Diff Content ===",
            "",
            record["git_diff"],
        ]

        with filepath.open("w", encoding="utf-8") as f:
            f.write("\n".join(content_lines))

    logging.info(f"Extracted {len(sampled_data)} diff files to {output_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Starting CCS dataset processing for concern extraction.")

    # Load SHA backup to exclude
    excluded_shas = load_sha_backup(SHA_BACKUP_PATH)

    # Load CCS dataset
    ccs_dataset = load_ccs_dataset(CCS_SOURCE_PATH)

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

    # Sample commits for each type
    all_sampled_data = []
    for commits in commits_by_type.values():
        sampled_data = sample_commits_for_type(
            commits, SAMPLES_PER_TYPE, OUTPUT_COLUMNS
        )
        all_sampled_data.extend(sampled_data)

    logging.info(f"Generated {len(all_sampled_data)} samples total")

    # Save to CSV
    save_to_csv(all_sampled_data, SAMPLED_CSV_PATH, OUTPUT_COLUMNS)

    # Extract diff files by type
    extract_diffs(all_sampled_data, DIFF_OUTPUT_DIR)

    # Append new SHAs to backup
    new_shas = [
        {"sha": record["sha"]} for record in all_sampled_data if record.get("sha")
    ]
    if new_shas:
        with SHA_BACKUP_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sha"])
            writer.writerows(new_shas)
        logging.info(f"Appended {len(new_shas)} new SHAs to backup")

    # Print summary
    type_counts = {}
    for record in all_sampled_data:
        commit_type = record.get("annotated_type", "")
        type_counts[commit_type] = type_counts.get(commit_type, 0) + 1

    print("\nSample distribution:")
    for commit_type in sorted(type_counts.keys()):
        print(f"  {commit_type}: {type_counts[commit_type]} samples")


if __name__ == "__main__":
    main()
