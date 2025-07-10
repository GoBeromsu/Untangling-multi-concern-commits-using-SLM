#!/usr/bin/env python3
"""
Convert tangled_dataset_with_reasoning.csv to JSON format.
Input: changes (diff)
Output: reasoning + commit_label in structured format
"""

import pandas as pd
import json
import ast
from pathlib import Path


def convert_csv_to_json(input_csv_path: str, output_json_path: str) -> None:
    """
    Convert CSV dataset to JSON format suitable for training.

    Args:
        input_csv_path: Path to input CSV file
        output_json_path: Path to output JSON file
    """
    print(f"Loading CSV from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    json_data = []

    for idx, row in df.iterrows():
        changes = row["changes"]
        types = row["types"]
        reasoning = row["reasoning"]

        # Parse types into a list if it's a string representation of a list
        if isinstance(types, str):
            try:
                # Try to parse as literal (for lists like "['fix']")
                commit_types = ast.literal_eval(types)
                if not isinstance(commit_types, list):
                    commit_types = [types]
            except (ValueError, SyntaxError):
                # If parsing fails, treat as single string
                commit_types = [types]
        else:
            commit_types = [str(types)]

        # Create the structured output format
        output_text = f"""<reasoning>
{reasoning}
</reasoning>

<commit_label>
{json.dumps(commit_types)}
</commit_label>"""

        # Create the training record
        record = {"prompt": changes, "answer": output_text}

        json_data.append(record)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} records...")

    print(f"Converting {len(json_data)} records to JSON...")

    # Write to JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Conversion complete! Saved to: {output_json_path}")

    # Print a sample record for verification
    print("\nSample record:")
    print(f"Prompt length: {len(json_data[0]['prompt'])} characters")
    print(f"Answer:\n{json_data[0]['answer']}")


def main():
    """Main function to run the conversion."""
    input_csv = "datasets/tangled_dataset_with_reasoning.csv"
    output_json = "datasets/tangled_dataset_training.json"

    # Ensure paths exist
    if not Path(input_csv).exists():
        print(f"Error: Input file not found: {input_csv}")
        return

    # Create output directory if needed
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    try:
        convert_csv_to_json(input_csv, output_json)
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()
