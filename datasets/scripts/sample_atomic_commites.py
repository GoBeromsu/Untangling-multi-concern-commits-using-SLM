import csv
import logging
import random
import pandas as pd
import tiktoken
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


CONVENTIONAL_COMMIT_TYPES = ["cicd"]
SAMPLES_PER_TYPE = 1
TARGET_TOKEN_LIMIT = 12288  # 16384 - 4096
ENCODING_MODEL = "cl100k_base"  # GPT-4 encoding
OUTPUT_COLUMNS = ["annotated_type", "masked_commit_message", "git_diff", "sha"]

# Path constants
CCS_SOURCE_PATH = Path("data/CCS Dataset Training Data.csv")
SAMPLED_CSV_PATH = Path("data/sampled_ccs_dataset.csv")
DIFF_OUTPUT_DIR = Path("data/types")


def load_existing_shas(file_path: Path) -> Set[str]:
    """Load existing SHAs from sampled dataset to exclude duplicates."""
    if not file_path.exists():
        logging.warning(f"Sampled dataset file not found: {file_path}")
        return set()

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        sha_set = set(df["sha"].astype(str))
        logging.info(f"Loaded {len(sha_set)} SHAs to exclude from existing samples")
        return sha_set
    except Exception as e:
        logging.error(f"Error loading existing SHAs: {e}")
        return set()


def load_ccs_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load CCS dataset CSV file and filter out commits exceeding token limit."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        df["annotated_type"] = (
            df["annotated_type"].str.lower().str.strip().replace("ci", "cicd")
        )
        data = df.to_dict("records")

        # Filter by token limit
        encoding = tiktoken.get_encoding(ENCODING_MODEL)
        filtered_data = []
        token_filtered_count = 0

        for item in data:
            # Only combine diff and commit message for token counting
            combined_text = f"{str(item.get('git_diff', ''))} {str(item.get('masked_commit_message', ''))}"
            token_count = len(encoding.encode(combined_text))

            if token_count <= TARGET_TOKEN_LIMIT:
                filtered_data.append(item)
            else:
                token_filtered_count += 1

        if token_filtered_count > 0:
            logging.info(
                f"Filtered out {token_filtered_count} commits exceeding {TARGET_TOKEN_LIMIT} tokens"
            )

        logging.info(
            f"Loaded {len(filtered_data)} records from CCS dataset (after token filtering)"
        )
        return filtered_data
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

    # Load existing SHAs from sampled dataset to exclude duplicates
    excluded_shas = load_existing_shas(SAMPLED_CSV_PATH)

    # Load CCS dataset
    ccs_dataset = load_ccs_dataset(CCS_SOURCE_PATH)

    # Filter out commits with SHAs in backup
    original_count = len(ccs_dataset)
    filtered_dataset = []
    for item in ccs_dataset:
        sha = str(item.get("sha", ""))
        if sha not in excluded_shas:
            filtered_dataset.append(item)
    ccs_dataset = filtered_dataset
    filtered_count = len(ccs_dataset)
    logging.info(
        f"Filtered out {original_count - filtered_count} commits already in existing samples"
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
