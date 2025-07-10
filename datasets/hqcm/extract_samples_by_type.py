#!/usr/bin/env python3
"""
Extract 30 samples per commit type from CodeFuse-HQCM dataset for manual processing.
Based on index.py logic but focused on sampling and saving individual type files.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any


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


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load CodeFuse-HQCM dataset JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def group_by_types(
    dataset: List[Dict[str, Any]], valid_types: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Groups commits by their concern type."""
    commits_by_type = {commit_type: [] for commit_type in valid_types}

    total_count = 0

    for item in dataset:
        total_count += 1
        commit_type = item.get("type")

        if commit_type in commits_by_type:
            commits_by_type[commit_type].append(item)

    print(f"Total commits: {total_count}")

    return commits_by_type


def sample_and_save_by_type(
    commits_by_type: Dict[str, List[Dict[str, Any]]],
    samples_per_type: int,
    output_dir: Path,
) -> None:
    """Sample random commits for each type and save to separate jsonl files."""
    output_dir.mkdir(exist_ok=True)

    total_sampled = 0

    for commit_type, commits in commits_by_type.items():
        if not commits:
            print(f"No commits found for type: {commit_type}")
            continue

        sampled_commits = random.sample(commits, samples_per_type)

        output_file = output_dir / f"{commit_type}_sample.jsonl"
        with output_file.open("w", encoding="utf-8") as f:
            for commit in sampled_commits:
                f.write(json.dumps(commit, ensure_ascii=False) + "\n")

        total_sampled += samples_per_type
        print(f"Saved {samples_per_type} samples for {commit_type} to {output_file}")

    print(f"Total samples saved: {total_sampled}")


def main() -> None:
    print("Starting manual process dataset generation.")

    # Set random seed for reproducibility
    random.seed(42)

    # Load and preprocess the raw dataset
    source_path = Path("codefuse-hqcm/dataset/train.json")
    hqcm_dataset = load_dataset(source_path)

    commits_by_type = group_by_types(hqcm_dataset, CONVENTIONAL_COMMIT_TYPES)

    # Print statistics
    print("\nCommits by type:")
    for commit_type, commits in commits_by_type.items():
        print(f"  {commit_type}: {len(commits)} commits")

    # Sample and save
    output_dir = Path("samples_by_type")
    sample_and_save_by_type(commits_by_type, SAMPLES_PER_TYPE, output_dir)

    print(f"\nGenerated {SAMPLES_PER_TYPE} samples per type for manual processing.")
    print(f"Files saved in: {output_dir}")


if __name__ == "__main__":
    main()
