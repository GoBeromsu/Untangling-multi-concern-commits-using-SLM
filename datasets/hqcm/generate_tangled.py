import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


TYPE_MAPPING = {
    "feat": "fe",
    "fix": "fi",
    "refactor": "re",
    "test": "te",
    "style": "st",
    "docs": "do",
    "build": "bu",
    "cicd": "ci",
}

PRIORITY_ORDER = ["ci", "bu", "do", "fe", "fi", "re", "te", "st"]


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load CodeFuse-HQCM dataset JSON file."""
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def generate_concern_filename(
    concerns: List[str], allow_same_types: bool, count: int, num_cases: int
) -> str:
    """Generate filename: c{N}_t{type_count}_s{sample_count}_{u|m}_{abbrev_list}"""
    abbrevs = [TYPE_MAPPING.get(concern, concern[:2]) for concern in concerns]
    label_type = "m" if allow_same_types else "u"

    ordered_abbrevs = sorted(
        abbrevs,
        key=lambda x: (
            PRIORITY_ORDER.index(x) if x in PRIORITY_ORDER else len(PRIORITY_ORDER)
        ),
    )

    return f"c{count}_t{len(concerns)}_s{num_cases}_{label_type}_{'_'.join(ordered_abbrevs)}"


def generate_tangled_cases(
    changes: List[Dict[str, Any]],
    allowed_types: List[str],
    concerns_per_case: int,
    num_cases: int,
    allow_same_types: bool = False,
) -> List[Dict[str, Any]]:
    """Generate tangled change cases."""
    filtered_changes = (
        [c for c in changes if c["type"] in allowed_types] if allowed_types else changes
    )

    changes_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for change in filtered_changes:
        changes_by_type[change["type"]].append(change)

    available_types = list(changes_by_type.keys())
    cases: List[Dict[str, Any]] = []
    seen_combinations = set()

    while len(cases) < num_cases:
        selected_types = (
            [random.choice(available_types) for _ in range(concerns_per_case)]
            if allow_same_types
            else random.sample(available_types, concerns_per_case)
        )

        atomic_changes = [
            random.choice(changes_by_type[selected_type])
            for selected_type in selected_types
        ]

        tangle_diff = "\n".join(change["change"] for change in atomic_changes)

        if tangle_diff in seen_combinations:
            continue

        seen_combinations.add(tangle_diff)
        cases.append(
            {
                "tangleChange": tangle_diff,
                "atomicChanges": [
                    {"change": change["change"], "label": change["type"]}
                    for change in atomic_changes
                ],
            }
        )

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
        default="cicd, feat",
        help="Comma-separated list of change types to include",
    )
    parser.add_argument(
        "--concerns",
        type=int,
        default=2,
        help="Number of concerns (atomic changes) per tangled case",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Number of tangled cases to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/tangled/codefuse-hqcm"),
        help="Directory to save the generated tangled cases JSON file",
    )
    parser.add_argument(
        "--allow-same-types",
        action="store_true",
        help="Allow same types in one tangled case",
    )

    args = parser.parse_args()
    random.seed(args.seed)
    allowed_types = [t.strip() for t in args.types.split(",") if t.strip()]

    filename = generate_concern_filename(
        allowed_types, args.allow_same_types, args.concerns, args.num_cases
    )
    output_path = args.output_dir / f"{filename}.json"

    changes = load_dataset(args.dataset_path)
    cases = generate_tangled_cases(
        changes, allowed_types, args.concerns, args.num_cases, args.allow_same_types
    )

    output_data = {
        "metadata": {
            "num_cases": len(cases),
            "concerns_per_case": args.concerns,
            "types": allowed_types,
            "allow_same_types": args.allow_same_types,
            "seed": args.seed,
            "source_dataset": str(args.dataset_path),
        },
        "cases": cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, fp=f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
