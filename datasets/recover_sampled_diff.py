#!/usr/bin/env python3
"""Advanced script to extract git changes from JSONL files by type."""

import json
import html
import os
import glob
from typing import Dict, Any, List


def decode_change_content(change_content: str) -> str:
    """Decode escaped change content from JSON to readable git diff format."""
    decoded = change_content.replace("\\n", "\n")
    decoded = html.unescape(decoded)
    return decoded


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    while "__" in filename:
        filename = filename.replace("__", "_")

    filename = filename.strip("_")[:100]
    return filename


def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of entries."""
    entries = []
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return entries


def get_all_sample_files(samples_dir: str) -> List[str]:
    """Get all JSONL sample files in directory."""
    pattern = os.path.join(samples_dir, "*.jsonl")
    return sorted(glob.glob(pattern))


def create_type_directories(base_output_dir: str, types: List[str]) -> Dict[str, str]:
    """Create directories for each type."""
    type_dirs = {}
    for type_name in types:
        type_dir = os.path.join(base_output_dir, type_name)
        os.makedirs(type_dir, exist_ok=True)
        type_dirs[type_name] = type_dir
    return type_dirs


def save_changes_by_type(
    entries: List[Dict[str, Any]], type_dirs: Dict[str, str]
) -> Dict[str, int]:
    """Save changes organized by type into separate directories."""
    type_counts = {}

    for entry in entries:
        change_type = entry.get("type", "unknown")
        subtype = entry.get("subtype", "")
        change_content = entry.get("change", "")
        summaries = entry.get("summaries", {})

        if change_type not in type_dirs:
            continue

        # Generate filename
        english_summary = summaries.get("en", "No summary")
        sanitized_summary = sanitize_filename(english_summary)

        # Count entries for this type
        if change_type not in type_counts:
            type_counts[change_type] = 0

        type_counts[change_type] += 1
        entry_num = type_counts[change_type]

        filename = f"{change_type}_{entry_num:03d}"
        if subtype:
            filename += f"_{subtype}"
        filename += f"_{sanitized_summary}.diff"

        filepath = os.path.join(type_dirs[change_type], filename)

        # Decode change content
        decoded_change = decode_change_content(change_content)

        # Create file content with metadata
        content_lines = [
            f"# Type: {change_type}",
            f"# Subtype: {subtype}",
            f"# English Summary: {summaries.get('en', 'N/A')}",
            f"# Chinese Summary: {summaries.get('zh', 'N/A')}",
            f"# Entry Number: {entry_num}",
            "",
            "# === Git Diff Content ===",
            "",
            decoded_change,
        ]

        try:
            with open(filepath, "w", encoding="utf-8") as file:
                file.write("\n".join(content_lines))
        except Exception as e:
            print(f"Error saving {filepath}: {e}")

    return type_counts


def display_summary(type_counts: Dict[str, int], total_processed: int):
    """Display summary of processed files."""
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total entries processed: {total_processed}")
    print("\nBreakdown by type:")

    for change_type, count in sorted(type_counts.items()):
        print(f"  {change_type:12}: {count:3d} files")

    print("=" * 60)


def main():
    """Main function to extract all changes by type."""
    print("Git Changes Extractor - Processing all files...")

    # Configuration
    samples_dir = "samples_by_type"
    base_output_dir = "extracted_changes_by_type"

    # Get all sample files
    sample_files = get_all_sample_files(samples_dir)

    if not sample_files:
        print(f"No JSONL files found in {samples_dir}")
        return

    print(f"Found {len(sample_files)} sample files")

    # Load all entries
    all_entries = []
    all_types = set()

    for sample_file in sample_files:
        entries = load_jsonl_file(sample_file)
        all_entries.extend(entries)

        for entry in entries:
            all_types.add(entry.get("type", "unknown"))

    print(f"Total entries: {len(all_entries)}")
    print(f"Types found: {sorted(all_types)}")

    # Create output directories
    type_dirs = create_type_directories(base_output_dir, sorted(all_types))

    # Group entries by type
    entries_by_type = {}
    for entry in all_entries:
        entry_type = entry.get("type", "unknown")
        if entry_type not in entries_by_type:
            entries_by_type[entry_type] = []
        entries_by_type[entry_type].append(entry)

    # Save all changes
    total_processed = 0
    type_counts = {}

    for entry_type, entries in entries_by_type.items():
        counts = save_changes_by_type(entries, {entry_type: type_dirs[entry_type]})
        type_counts.update(counts)
        total_processed += len(entries)

    # Display summary
    display_summary(type_counts, total_processed)
    print(f"\nAll files saved to: {base_output_dir}")
    print("Process completed! 🎉")


if __name__ == "__main__":
    main()
