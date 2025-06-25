import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import sys


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load CodeFuse-HQCM dataset JSON file.

    Each dataset file contains a list of change objects with at least the keys
    `type` and `change`.
    """
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def generate_tangled_cases(
    changes: List[Dict[str, Any]],
    allowed_types: List[str],
    concerns_per_case: int,
    num_cases: int,
    ensure_different_types: bool = True,
) -> List[Dict[str, Any]]:
    """Generate tangled change cases.

    Parameters
    ----------
    changes: List of atomic change dicts from the dataset.
    allowed_types: List of commit *types* that are eligible for sampling.
    concerns_per_case: Number of atomic changes (concerns) to bundle into one tangled change.
    num_cases: Number of tangled cases to create.
    ensure_different_types: If True, ensure each tangled case contains changes of different types.
    """
    # Filter changes by allowed types if provided (ignore empty list means all)
    filtered_changes = (
        [c for c in changes if c["type"] in allowed_types] if allowed_types else changes
    )
    if not filtered_changes:
        raise ValueError("No changes match the specified types.")

    # Group changes by type
    changes_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for change in filtered_changes:
        changes_by_type[change["type"]].append(change)

    available_types = list(changes_by_type.keys())

    # Validation based on ensure_different_types
    if ensure_different_types:
        if len(available_types) < concerns_per_case:
            raise ValueError(
                f"Need at least {concerns_per_case} different types for each case, "
                f"but only {len(available_types)} types available: {available_types}"
            )

        min_changes_per_type = min(
            len(changes_list) for changes_list in changes_by_type.values()
        )
        if min_changes_per_type * len(available_types) < concerns_per_case * num_cases:
            max_possible_cases = (
                min_changes_per_type * len(available_types) // concerns_per_case
            )
            raise ValueError(
                f"Cannot generate {num_cases} cases. Maximum possible with different types: {max_possible_cases}"
            )
    else:
        total_required = concerns_per_case * num_cases
        if total_required > len(filtered_changes):
            raise ValueError(
                f"Requested {total_required} atomic changes ({concerns_per_case} concerns x {num_cases} cases) "
                f"but only {len(filtered_changes)} available."
            )

    # Shuffle changes within each type
    for type_changes in changes_by_type.values():
        random.shuffle(type_changes)

    # Keep track of used changes per type
    used_counts = {type_name: 0 for type_name in available_types}

    cases: List[Dict[str, Any]] = []
    for _ in range(num_cases):
        atomic_changes: List[Dict[str, Any]] = []

        if ensure_different_types:
            # Select different types for this case
            selected_types = random.sample(available_types, concerns_per_case)
            for selected_type in selected_types:
                change = changes_by_type[selected_type][used_counts[selected_type]]
                atomic_changes.append(change)
                used_counts[selected_type] += 1
        else:
            # Allow selecting from the same type multiple times
            for _ in range(concerns_per_case):
                # Select a random type (can be repeated)
                selected_type = random.choice(available_types)
                change = changes_by_type[selected_type][used_counts[selected_type]]
                atomic_changes.append(change)
                used_counts[selected_type] += 1

        # Create tangled case
        tangle_diff = "\n".join(change["change"] for change in atomic_changes)
        case = {
            "tangleChange": tangle_diff,
            "atomicChanges": [
                {"change": change["change"], "label": change["type"]}
                for change in atomic_changes
            ],
        }
        cases.append(case)

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tangled change JSON cases from CodeFuse-HQCM dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("datasets/candidates/codefuse-hqcm/dataset/test.json"),
        help="Path to the dataset JSON file (train.json or test.json)",
    )
    parser.add_argument(
        "--types",
        type=str,
        default="test,build,cicd,docs",  # top 4 types easily distinguishable by LLM
        help="Comma-separated list of change types to include. Leave empty for all types.",
    )
    parser.add_argument(
        "--concerns",
        type=int,
        default=2,
        help="Number of concerns (atomic changes) per tangled case.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Number of tangled cases to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("datasets/tangled/codefuse-hqcm/tangled_cases.json"),
        help="Path to save the generated tangled cases JSON file.",
    )
    parser.add_argument(
        "--ensure-different-types",
        action="store_true",
        default=True,
        help="Ensure each tangled case contains changes of different types (default: True).",
    )
    parser.add_argument(
        "--allow-same-types",
        action="store_true",
        default=False,
        help="Allow same types in one tangled case (overrides --ensure-different-types).",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    allowed_types = [t.strip() for t in args.types.split(",") if t.strip()]

    # Handle conflicting options
    ensure_different_types = args.ensure_different_types and not args.allow_same_types

    changes = load_dataset(args.dataset_path)
    cases = generate_tangled_cases(
        changes, allowed_types, args.concerns, args.num_cases, ensure_different_types
    )

    # Create output structure with metadata
    output_data = {
        "metadata": {
            "num_cases": len(cases),
            "concerns_per_case": args.concerns,
            "types": allowed_types,
            "ensure_different_types": ensure_different_types,
            "seed": args.seed,
            "source_dataset": str(args.dataset_path),
        },
        "cases": cases,
    }

    # Create output directory if it doesn't exist
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, fp=f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
