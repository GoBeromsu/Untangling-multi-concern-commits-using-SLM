import os
import sys
import json
from pathlib import Path
from typing import Dict
import tiktoken
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from utils.llms import constant
import utils.llms as llms
import utils.eval as eval_utils
import utils.prompt as prompt_utils

# Data key constants
COMMIT_MESSAGE_KEY = "commit_message"
GIT_DIFF_KEY = "git_diff"
MASKED_COMMIT_MESSAGE_KEY = "masked_commit_message"
SHA_KEY = "sha"
SHAS_KEY = "shas"
TYPES_KEY = "types"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

CONTEXT_WINDOW = [1024, 2048, 4096, 8192, 12288]

# contextWindow = [1024]
MODEL_NAMES = [
    # "microsoft/phi-4",  # LM Studio model
    # "gpt-4o-mini",  # OpenAI model
    "gpt-4.1-2025-04-14",
]
encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding


def truncate_commits(
    commits: Dict[str, Dict[str, str]],
    context_window: int,
    include_message: bool = True,
) -> str:
    concern_count = len(commits)
    available_tokens_per_commit = context_window // concern_count

    messages, diffs = [], []

    for commit_data in commits.values():
        message = commit_data[COMMIT_MESSAGE_KEY]
        diff = commit_data[GIT_DIFF_KEY]

        if include_message:
            message_tokens = encoding.encode(message)
            remaining_tokens = available_tokens_per_commit - len(message_tokens)
            messages.append(message)
        else:
            remaining_tokens = available_tokens_per_commit

        diff_tokens = encoding.encode(diff)
        truncated_diff = (
            diff
            if len(diff_tokens) <= remaining_tokens
            else encoding.decode(diff_tokens[:remaining_tokens])
        )

        diffs.append(truncated_diff)

    if include_message:
        return f"Commit Message: {' '.join(messages)}\nDiff: {' '.join(diffs)}"
    else:
        return f"Diff: {' '.join(diffs)}"


def measure(
    model_name,
    prompt_type,
    context_window,
    commit_type,
    tangled_df,
    atomic_df,
    prompt_dir,
):
    print(
        f"\n=== Model: {model_name}, Prompt Type: {commit_type}, Prompt: {prompt_type}, Context Window: {context_window} ==="
    )

    file_name = f"{model_name.replace('/', '_')}_{prompt_type.replace('-', '_')}_{context_window}.csv"
    csv_path = prompt_dir / file_name
    create_csv_with_headers(csv_path)

    for idx, row in tangled_df.iterrows():
        shas = json.loads(row[SHAS_KEY])
        commits: Dict[str, str] = {}
        for sha in shas:
            matching_row = atomic_df[atomic_df[SHA_KEY] == sha]
            commit_message = matching_row[MASKED_COMMIT_MESSAGE_KEY].values[0]
            git_diff = matching_row[GIT_DIFF_KEY].values[0]
            commits[sha] = {
                COMMIT_MESSAGE_KEY: commit_message,
                GIT_DIFF_KEY: git_diff,
            }

        system_prompt = prompt_utils.get_prompt_by_type(prompt_type, include_message)
        truncated_commit = truncate_commits(commits, context_window, include_message)

        try:
            api_call = lambda: llms.api_call(
                model_name=model_name,
                commit=truncated_commit,
                system_prompt=system_prompt,
                api_key=OPENAI_KEY,
            )
            predicted_types, inference_time = eval_utils.measure_inference_time(
                api_call
            )
        except Exception as e:
            print(f"Unexpected error processing row {idx}: {e}")
            predicted_types = []
            inference_time = 0.0

        result = {column: None for column in constant.DEFAULT_DF_COLUMNS}
        result["predicted_types"] = predicted_types
        result["actual_types"] = json.loads(row[TYPES_KEY])
        result["inference_time"] = inference_time
        result[SHAS_KEY] = shas

        result_df = pd.DataFrame([result])
        result_df.to_csv(csv_path, mode="a", header=False, index=False)

        print(
            f"Row {idx}: inference_time(sec): {inference_time:.2f} - saved to {csv_path}"
        )


def create_csv_with_headers(csv_path: Path) -> None:
    """Create CSV file with headers if it doesn't exist."""
    if not csv_path.exists():
        df = pd.DataFrame(columns=constant.DEFAULT_DF_COLUMNS)
        df.to_csv(csv_path, index=False)


def main():
    tangled_df = eval_utils.load_dataset("test")
    atomic_df = eval_utils.load_dataset("atomic")

    commit_types = ["with_message", "diff_only"]
    prompt_types = ["Zero-shot", "One-shot", "Two-shot"]
    for model_name in MODEL_NAMES:
        print(f"\n{'='*50}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*50}")

        for commit_type in commit_types:
            prompt_dir = Path("results") / commit_type
            prompt_dir.mkdir(parents=True, exist_ok=True)

            # Determine if commit message should be included based on prompt type
            include_message = commit_type == "with_message"

            for prompt_type in prompt_types:
                for context_window in CONTEXT_WINDOW:
                    measure(
                        model_name,
                        prompt_type,
                        context_window,
                        commit_type,
                        tangled_df,
                        atomic_df,
                        prompt_dir,
                    )

        llms.clear_cache()

    print(f"\nDataset Summary:")
    print(f"Loaded tangled dataset: {len(tangled_df)} samples")
    print(f"Loaded atomic dataset: {len(atomic_df)} samples")
    print(f"Processed models: {', '.join(MODEL_NAMES)}")
    print(f"Results saved in results/ directory, organized by prompt type")


if __name__ == "__main__":
    main()
