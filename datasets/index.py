import json
import csv
import logging
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
DATASET_SIZE = 6000
EXAMPLES_PER_CONCERN_COUNT = DATASET_SIZE // 3  # 2000 examples each


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load CodeFuse-HQCM dataset JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_to_csv(data: List[Dict[str, str]], output_path: Path) -> None:
    """Save processed data to CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


def group_commits_by_type(
    dataset: List[Dict[str, Any]], valid_types: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Groups commits by their concern type, handling style->refactor mapping."""
    commits_by_type = {commit_type: [] for commit_type in valid_types}

    for item in dataset:
        commit_type = item.get("type")
        if commit_type == "style":
            commit_type = "refactor"

        commits_by_type[commit_type].append(item)

    print(f"Total number of commits: {sum(len(v) for v in commits_by_type.values())}")
    return commits_by_type


def sample_one_concern(
    commits_by_type: Dict[str, List[Dict[str, Any]]]
) -> tuple[str, str]:
    """Uniformly samples one concern type and returns (type, change)."""
    types = list(commits_by_type.keys())
    chosen_type = random.choice(types)
    chosen_change = random.choice(commits_by_type[chosen_type])["change"]
    return chosen_type, chosen_change


def sample_concerns_freq_aware(
    commits_by_type: Dict[str, List[Dict[str, Any]]],
    num_examples: int,
    concern_count: int,
) -> List[Dict[str, str]]:
    """Creates examples by independently sampling each concern, excluding previously selected types."""
    examples = []

    for _ in range(num_examples):
        selected_types = []
        selected_changes = []
        available_commits = commits_by_type.copy()

        # Sample concern_count times, excluding previously selected types
        for _ in range(concern_count):

            concern_type, change = sample_one_concern(available_commits)
            selected_types.append(concern_type)
            selected_changes.append(change)

            # Remove the selected type from available options
            del available_commits[concern_type]

        examples.append(
            {
                "changes": "\n".join(selected_changes),
                "types": ",".join(sorted(selected_types)),
                "count": str(concern_count),
            }
        )

    return examples


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Starting dataset generation with uniform probability sampling.")

    # Load and preprocess the raw dataset
    source_path = Path("datasets/codefuse-hqcm/dataset/train.json")
    hqcm_dataset = load_dataset(source_path)

    commits_by_type = group_commits_by_type(hqcm_dataset, CONVENTIONAL_COMMIT_TYPES)

    logging.info(
        f"Loaded and grouped {sum(len(c) for c in commits_by_type.values())} commits by type."
    )

    # Generate examples for each concern count (1000 each)
    examples = []
    examples.extend(
        sample_concerns_freq_aware(commits_by_type, EXAMPLES_PER_CONCERN_COUNT, 1)
    )
    examples.extend(
        sample_concerns_freq_aware(commits_by_type, EXAMPLES_PER_CONCERN_COUNT, 2)
    )
    examples.extend(
        sample_concerns_freq_aware(commits_by_type, EXAMPLES_PER_CONCERN_COUNT, 3)
    )
    logging.info(
        f"Generated {len(examples)} examples (1000 each for 1, 2, 3 concerns)."
    )

    # Count occurrences of each type in the generated examples
    type_counts = {}
    for example in examples:
        types = example["types"]
        type_counts[types] = type_counts.get(types, 0) + 1

    print("Type distribution in generated examples:")
    for types in sorted(type_counts.keys()):
        print(f"  {types}: {type_counts[types]}")

    # Save single CSV file
    output_dir = Path("datasets")
    output_dir.mkdir(exist_ok=True)

    save_to_csv(examples, output_dir / "tangled_dataset.csv")


if __name__ == "__main__":
    main()