#!/usr/bin/env python3
"""
A script to verify tangled concerns from a CSV file using OpenAI's API with concern count hints.

This script reads tangled commits from a CSV file, sends the diff of each commit
to the OpenAI API with the expected number of concerns, and then compares the 
predicted concerns with the ground truth from the CSV file.
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
MODEL = "gpt-4o-mini"
INPUT_CSV_PATH = "datasets/tangled/tangled_Zeebe.csv"
MODEL_CONTEXT_LIMIT = 128000
# Add a buffer for the system prompt and other overhead
TOKEN_LIMIT = MODEL_CONTEXT_LIMIT - 1000
q
COMMIT_TYPES = ["feat", "fix", "perf", "style", "refactor", "docs", "test", "ci", "build", "chore"]

def create_system_prompt(concern_count: int) -> str:
    """
    Create a system prompt with the expected number of concerns.
    
    Args:
        concern_count: The number of concerns expected in the diff
        
    Returns:
        The formatted system prompt string
    """
    return f"""Extract and classify commit concerns from the given code diff. This diff contains exactly {concern_count} tangled concerns that need to be identified and classified according to conventional commit types.

Please classify each identified concern into one of the ten categories: feat, fix, perf, style, refactor, docs, test, ci, build, and chore.

**Detailed Category Definitions:**

- **feat**: Code changes that introduce new functionality, including internal or user-facing features. This encompasses additions that enhance the capabilities of the software system and expand its functionality.

- **fix**: Code changes that resolve faults or bugs within the codebase. These modifications address errors that affect correct behavior and system stability.

- **perf**: Code changes that improve performance, such as enhancing execution speed, reducing memory consumption, or optimizing resource utilization. Focus on measurable performance improvements.

- **style**: Code changes that improve readability without affecting the meaning or behavior of the code. This encompasses variable naming improvements, indentation fixes, formatting adjustments, and addressing linting or code analysis warnings.

- **refactor**: Code changes that restructure the program without changing its external behavior, aiming to improve maintainability, modularity, or scalability. This category explicitly excludes changes classified as "perf" or "style". Examples include enhancing modularity, refining exception handling, improving scalability, conducting code cleanup, and removing deprecated code.

- **docs**: Code changes that modify documentation or text content, such as correcting typos in comments, updating inline documentation, modifying README files, or updating API documentation.

- **test**: Code changes that modify test files, including the addition of new tests, updating existing test cases, or improving test coverage and quality.

- **ci**: Code changes to Continuous Integration configuration files and scripts, such as configuring or updating CI/CD pipelines, modifying workflow files (e.g., `.travis.yml`, `.github/workflows`), or adjusting deployment scripts.

- **build**: Code changes affecting the build system or dependency management (e.g., Maven, Gradle, Cargo, npm). Examples include updating dependencies, configuring build configurations, adding build scripts, or modifying package management files.

- **chore**: Code changes for miscellaneous tasks that do not neatly fit into any of the above categories. This includes auxiliary maintenance tasks, tooling updates, or other non-functional changes.

**Instructions:**
1. Analyze the entire diff carefully to identify exactly {concern_count} distinct concerns present
2. Each concern should be classified into exactly one of the above categories
3. Return only the identified types in JSON format
4. Do not provide explanations or additional commentary

Return JSON only with the structure: {{"types": ["type1", "type2", ...]}}"""

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


def get_predicted_types(diff: str, concern_count: int, headers: dict) -> Set[str]:
    """
    Sends a diff to the OpenAI API via REST and returns the predicted concern types.

    Args:
        diff: The code diff to be analyzed.
        concern_count: The expected number of concerns in the diff.
        headers: The request headers with Authorization.

    Returns:
        A set of predicted concern types.
    """
    system_prompt = create_system_prompt(concern_count)
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
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
        response.raise_for_status()
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
    Main function to run the verification process with concern count hints.
    """
    logger.info("Starting concern verification script with count hints.")
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

    # Verify required columns exist
    required_columns = ["repository_name", "types", "type_count", "diff"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return

    correct_predictions = 0
    total_predictions = 0

    for index, row in df.iterrows():
        diff = row.get("diff", "")
        ground_truth_types_str = row.get("types", "")
        type_count = row.get("type_count", 0)
        repository_name = row.get("repository_name", "unknown")

        if not diff or not ground_truth_types_str or not type_count:
            logger.warning(f"Skipping row {index} due to missing 'diff', 'types', or 'type_count'.")
            continue

        token_count = len(encoding.encode(diff))
        if token_count > TOKEN_LIMIT:
            logger.warning(f"Skipping record {index} due to excessive token count: {token_count} > {TOKEN_LIMIT}")
            continue

        ground_truth_types = set(ground_truth_types_str.split(','))
        predicted_types = get_predicted_types(diff, type_count, headers)

        is_match = (ground_truth_types == predicted_types)
        total_predictions += 1
        if is_match:
            correct_predictions += 1

        logger.info(f"--- Record {index} ({repository_name}) ---")
        logger.info(f"Expected Count: {type_count}")
        logger.info(f"Ground Truth:   {sorted(list(ground_truth_types))}")
        logger.info(f"Predicted:      {sorted(list(predicted_types))}")
        logger.info(f"Match:          {'✅ Correct' if is_match else '❌ Incorrect'}")
        logger.info("-" * (20 + len(str(index)) + len(repository_name)))

    # Final accuracy report
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions * 100
        logger.info(f"\n{'='*50}")
        logger.info(f"FINAL RESULTS:")
        logger.info(f"Total predictions: {total_predictions}")
        logger.info(f"Correct predictions: {correct_predictions}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"{'='*50}")


if __name__ == "__main__":
    main() 