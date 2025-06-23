#!/usr/bin/env python3
"""
A simple script to verify tangled concerns from a CSV file using OpenAI's API.

This script reads tangled commits from a CSV file, sends the diff of each commit
to the OpenAI API (using a model like gpt-4o-mini) to identify the
commit's concerns, and then compares the predicted concerns with the ground
truth from the CSV file.
"""

import json
import logging
import os
from typing import Set

import pandas as pd
import requests
from dotenv import load_dotenv
import tiktoken

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---
MODEL = "gpt-4-turbo"
INPUT_CSV_PATH = "datasets/tangled/tangled_Zeebe.csv"
MODEL_CONTEXT_LIMIT = 128000
# Add a buffer for the system prompt and other overhead
TOKEN_LIMIT = MODEL_CONTEXT_LIMIT - 1000

COMMIT_TYPES = ["feat", "fix", "perf", "style", "refactor", "docs", "test", "ci", "build", "chore"]
SYSTEM_PROMPT = """Extract commit concerns and classify each with conventional commit types.

Types:
- feat: Code changes that introduce new functionality, including internal or user-facing features. This includes additions that enhance the capabilities of the software system.
- fix: Code changes that resolve faults or bugs. These modifications address errors that affect correct behaviour.
- perf: Code changes that optimise performance, such as improvements in execution speed or memory efficiency.
- style: Code changes that improve readability or adhere to formatting standards, without affecting the logic or meaning. Includes naming, indentation, or linting adjustments.
- refactor: Code changes that restructure code to improve maintainability, modularity, or scalability without changing its external behaviour. This excludes "perf" and "style" changes. Examples: code cleanup, exception handling improvements, deprecated code removal.
- docs: Code changes that affect documentation. Includes comment updates, typo corrections, and documentation file changes.
- test: Code changes that modify test files, including test additions or updates.
- ci: Code changes to Continuous Integration configuration or workflow scripts (e.g., .travis.yml, .github/workflows).
- build: Code changes to the build system or dependencies. Includes build configuration files, dependency upgrades, or build scripts.
- chore: Code changes that do not fit into other categories. Includes auxiliary or maintenance tasks.

Return JSON only.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "types": {
            "type": "array",
            "items": {"type": "string", "enum": COMMIT_TYPES},
        }
    },
    "required": ["types"],
    "additionalProperties": False,
}


def get_predicted_types(diff: str, headers: dict) -> Set[str]:
    """
    Sends a diff to the OpenAI API via REST and returns the predicted concern types.

    Args:
        diff: The code diff to be analyzed.
        headers: The request headers with Authorization.

    Returns:
        A set of predicted concern types.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Tangled code changes diff:\n{diff}"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "extract_concern_types",
                "description": "Extracts the conventional commit types from a code diff.",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            },
        },
        "temperature": 0,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        if content:
            structured_output = json.loads(content)
            return set(structured_output.get("types", []))
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while calling OpenAI API via REST: {e}")
        if hasattr(e, 'response') and e.response is not None:
             logger.error(f"Error Response Body: {e.response.text}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse API response: {e}")
        if 'response' in locals() and response is not None:
             logger.error(f"Received response text: {response.text}")

    return set()


def main():
    """
    Main function to run the verification process.
    """
    logger.info("Starting concern verification script.")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        encoding = tiktoken.encoding_for_model(MODEL)
    except KeyError:
        logger.info("Model not found for tiktoken, using cl100k_base.")
        encoding = tiktoken.get_encoding("cl100k_base")

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        logger.info(f"Successfully loaded {len(df)} records from {INPUT_CSV_PATH}")
    except FileNotFoundError:
        logger.error(f"Input file not found at {INPUT_CSV_PATH}")
        return

    for index, row in df.iterrows():
        diff = row.get("diff", "")
        ground_truth_types_str = row.get("types", "")

        if not diff or not ground_truth_types_str:
            logger.warning(f"Skipping row {index} due to missing 'diff' or 'types'.")
            continue

        token_count = len(encoding.encode(diff))
        if token_count > TOKEN_LIMIT:
            logger.warning(f"Skipping record {index} due to excessive token count: {token_count} > {TOKEN_LIMIT}")
            continue

        ground_truth_types = set(ground_truth_types_str.split(','))
        predicted_types = get_predicted_types(diff, headers)

        is_match = (ground_truth_types == predicted_types)

        logger.info(f"--- Record {index} ---")
        logger.info(f"Ground Truth: {sorted(list(ground_truth_types))}")
        logger.info(f"Predicted:    {sorted(list(predicted_types))}")
        logger.info(f"Match:        {'✅ Correct' if is_match else '❌ Incorrect'}")
        logger.info("-" * (16 + len(str(index))))


if __name__ == "__main__":
    main() 