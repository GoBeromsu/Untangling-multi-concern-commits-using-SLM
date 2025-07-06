"""
Validation logic with Named Conditions following frontend design principles.
"""

import os

def is_openai_api_key_available() -> bool:
    """Check if OpenAI API key is configured and available."""
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key is not None and api_key.strip() != ""


def is_valid_dataset_file(file_path: str) -> bool:
    """Check if file is a valid dataset file (JSON format, not candidate)."""
    is_json_file = file_path.endswith(".json")
    is_not_candidate_file = "candidate" not in file_path
    return is_json_file and is_not_candidate_file
