#!/usr/bin/env python3
"""
Add reasoning column to tangled_css_dataset.csv using OpenAI API.
Follows established patterns from openai.py with structured output and proper error handling.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import openai
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration constants
DATASET_FILE_PATH: Path = Path("data/tangled_css_dataset.csv")
OUTPUT_FILE_PATH: Path = Path("data/tangled_css_dataset_with_reasoning.csv")
OPENAI_MODEL: str = "gpt-4.1-2025-04-14"
REASONING_TEMPERATURE: float = 0.7
REQUEST_DELAY_SECONDS: float = 0.5
BATCH_SIZE: int = 10

# Reasoning response schema for structured output
REASONING_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Detailed reasoning explaining the classification process and decision-making steps",
        }
    },
    "required": ["reasoning"],
    "additionalProperties": False,
}

# Structured output format for reasoning generation
REASONING_STRUCTURED_OUTPUT_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "reasoning_generation_response",
        "schema": REASONING_RESPONSE_SCHEMA,
        "strict": True,
    },
}

# System prompt for reasoning generation
SYSTEM_PROMPT: str = """You are a software engineer classifying individual code units extracted from a tangled commit.
Each change unit (e.g., function, method, class, or code block) represents a reviewable atomic change, and must be assigned exactly one label.

Label selection must assign exactly one concern from the following unified set:
- Purpose labels: the motivation behind making a code change (feat, fix, refactor)
- Object labels: the essence of the code changes that have been made (docs, test, cicd, build)
  - Use an object label only when the code unit is fully dedicated to that artifact category (e.g., writing test logic, modifying documentation).

# Instructions
1. Review the code unit and determine the most appropriate label from the unified set.
2. If multiple labels seem possible, resolve the overlap by applying the following rule:
   - **Purpose + Purpose**: Choose the label that best reflects *why* the change was made — `fix` if resolving a bug, `feat` if adding new capability, `refactor` if improving structure without changing behavior.
   - **Object + Object**: Choose the label that reflects the *functional role* of the artifact being modified — e.g., even if changing build logic, editing a CI script should be labeled as `cicd`.
   - **Purpose + Object**: If the change is driven by code behavior (e.g., fixing test logic), assign a purpose label; if it is entirely scoped to a support artifact (e.g., adding new tests), assign an object label.

# Labels
- feat: Introduces new features to the codebase.
- fix: Fixes bugs or faults in the codebase.
- refactor: Restructures existing code without changing external behavior (e.g., improves readability, simplifies complexity, removes unused code).
- docs: Modifies documentation or text (e.g., fixes typos, updates comments or docs).
- test: Modifies test files (e.g., adds or updates tests).
- cicd: Updates CI (Continuous Integration) configuration files or scripts (e.g., `.travis.yml`, `.github/workflows`).
- build: Affects the build system (e.g., updates dependencies, changes build configs or scripts).

Based on the provided commit message, diff, and assigned types, generate a detailed reasoning that explains how you would apply the above instructions to classify each code unit in this tangled commit. Focus on describing the classification process and decision-making steps."""


def load_dataset_from_path(file_path: Path) -> pd.DataFrame:
    """Load dataset from CSV file path."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset from {file_path}: {e}")
        raise


def authenticate_openai_client() -> openai.OpenAI:
    """Authenticate with OpenAI API and return client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return openai.OpenAI(api_key=api_key)


def create_reasoning_prompt(description: str, diff: str, types: str) -> str:
    """Create user prompt for reasoning generation."""
    try:
        types_list = json.loads(types)
        types_formatted = ", ".join(types_list)
    except json.JSONDecodeError:
        types_formatted = types

    return f"""Commit Message: {description}

Assigned Types: {types_formatted}

Git Diff:
{diff}

Please generate detailed reasoning explaining how you would apply the classification instructions to analyze this tangled commit."""


def generate_reasoning_with_openai(
    client: openai.OpenAI, description: str, diff: str, types: str, row_index: int
) -> str:
    """Generate reasoning using OpenAI API with structured output."""
    try:
        user_prompt = create_reasoning_prompt(description, diff, types)

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=REASONING_TEMPERATURE,
            response_format=REASONING_STRUCTURED_OUTPUT_FORMAT,
        )

        content = response.choices[0].message.content
        if not content:
            logging.warning(f"Empty response from OpenAI for row {row_index}")
            return "No reasoning generated"

        # Parse structured response
        try:
            response_data = json.loads(content)
            reasoning = response_data.get("reasoning", "No reasoning in response")
            logging.info(f"Generated reasoning for row {row_index}")
            return reasoning.strip()
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse structured response for row {row_index}")
            return content.strip()

    except openai.APIError as e:
        logging.error(f"OpenAI API error for row {row_index}: {e}")
        return f"API Error: {e}"
    except Exception as e:
        logging.error(f"Unexpected error for row {row_index}: {e}")
        return f"Error: {e}"


def add_reasoning_column_to_dataset(
    df: pd.DataFrame, client: openai.OpenAI
) -> pd.DataFrame:
    """Add reasoning column to dataset using OpenAI API with sequential index assignment."""
    total_rows = len(df)

    # Initialize reason column with empty values
    df["reason"] = ""

    logging.info(f"Starting reasoning generation for {total_rows} rows...")

    for idx, row in df.iterrows():
        description = str(row["description"])
        diff = str(row["diff"])
        types = str(row["types"])

        logging.info(f"Processing row {idx + 1}/{total_rows} (index: {idx})")

        reasoning = generate_reasoning_with_openai(
            client, description, diff, types, idx + 1
        )

        # Assign reasoning directly to the specific index to prevent mismatch
        df.at[idx, "reason"] = reasoning

        # Log assignment confirmation
        logging.info(f"Assigned reasoning to index {idx}: {reasoning[:50]}...")

        # Rate limiting delay
        time.sleep(REQUEST_DELAY_SECONDS)

        # Progress reporting
        if (idx + 1) % BATCH_SIZE == 0:
            logging.info(f"Completed {idx + 1}/{total_rows} rows")

    logging.info("Successfully added reasoning column to dataset")
    return df


def save_dataset_to_path(df: pd.DataFrame, output_path: Path) -> None:
    """Save dataset to CSV file path."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logging.info(f"Saved dataset to {output_path}")
        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")

    except Exception as e:
        logging.error(f"Error saving dataset to {output_path}: {e}")
        raise


def validate_environment_setup() -> None:
    """Validate environment configuration and file availability."""
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Check if dataset file exists
    if not DATASET_FILE_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE_PATH}")


def configure_logging() -> None:
    """Configure logging with structured format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def process_reasoning_generation(
    input_file_path: Path, output_file_path: Path, test_mode: bool = False
) -> None:
    """Process reasoning generation for dataset."""
    # Load dataset
    df = load_dataset_from_path(input_file_path)

    # Check if reason column already exists
    if "reason" in df.columns:
        if test_mode:
            logging.info("Reason column exists in test mode, dropping for regeneration")
            df = df.drop("reason", axis=1)
        else:
            logging.warning("Reason column already exists in dataset")
            response = input("Do you want to regenerate all reasoning? (y/n): ")
            if response.lower() != "y":
                logging.info("Exiting without changes")
                return
            df = df.drop("reason", axis=1)

    # Authenticate OpenAI client
    client = authenticate_openai_client()

    # Generate reasoning
    df_with_reasoning = add_reasoning_column_to_dataset(df, client)

    # Save results
    save_dataset_to_path(df_with_reasoning, output_file_path)


def main() -> None:
    """Main function to add reasoning column to dataset."""
    configure_logging()

    logging.info("Starting reasoning column addition process...")

    try:
        # Determine if running in test mode
        test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"

        # Set dataset file path based on mode
        if test_mode:
            logging.info("Running in test mode with sample data")
            global DATASET_FILE_PATH
            DATASET_FILE_PATH = Path("../data/tangled_test_sample.csv")
            output_path = Path("../data/tangled_test_sample_with_reasoning.csv")
        else:
            output_path = OUTPUT_FILE_PATH

        # Validate environment after path is set
        validate_environment_setup()

        logging.info(f"Input: {DATASET_FILE_PATH}")
        logging.info(f"Output: {output_path}")
        logging.info(f"Model: {OPENAI_MODEL}")
        logging.info(f"Temperature: {REASONING_TEMPERATURE}")

        process_reasoning_generation(DATASET_FILE_PATH, output_path, test_mode)

        logging.info("Reasoning column addition completed successfully!")

    except Exception as e:
        logging.error(f"Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
