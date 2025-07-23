import sys
import json
from pathlib import Path
from typing import Dict, List
import tiktoken
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import utils.llms as llms
import utils.eval as eval_utils
import utils.prompt as prompt_utils

# contextWindow = [1024, 2048, 4096, 8192, 12288]
contextWindow = [1024]
model_names = ["microsoft/phi-4"]
encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding


def truncate_commits(commits: Dict[str, Dict[str, str]], context_window: int) -> str:
    concern_count = len(commits)
    available_tokens_per_commit = context_window // concern_count

    messages, diffs = [], []

    for commit_data in commits.values():
        message = commit_data["commit_message"]
        diff = commit_data["git_diff"]

        message_tokens = encoding.encode(message)
        remaining_tokens = available_tokens_per_commit - len(message_tokens)

        diff_tokens = encoding.encode(diff)
        truncated_diff = (
            diff
            if len(diff_tokens) <= remaining_tokens
            else encoding.decode(diff_tokens[:remaining_tokens])
        )

        messages.append(message)
        diffs.append(truncated_diff)

    return f"{'Commit Message: ' + '\n'.join(messages)}\n{'Diff: ' + '\n'.join(diffs)}"


def create_csv_with_headers(csv_path: Path) -> None:
    """Create CSV file with headers if it doesn't exist."""
    if not csv_path.exists():
        headers = [
            "row_index",
            "context_window",
            "predicted_types",
            "actual_types",
            "inference_time",
            "shas",
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(csv_path, index=False)


def setup_model(model_name: str) -> bool:
    """Setup and validate LM Studio model."""
    available_models, _ = llms.get_models()
    if not available_models or model_name not in available_models:
        return False

    try:
        llms.load_model(model_name)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    tangled_df = eval_utils.load_dataset("test")
    atomic_df = eval_utils.load_dataset("atomic")

    prompt_types = [
        ("with_message", prompt_utils.get_system_prompt_with_message),
        ("diff_only", prompt_utils.get_system_prompt_diff_only),
    ]

    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*50}")

        if not setup_model(model_name):
            print(f"Failed to setup model: {model_name}")
            continue

        for prompt_type, get_prompt in prompt_types:
            prompt_dir = Path("results") / prompt_type
            prompt_dir.mkdir(parents=True, exist_ok=True)

            for context_window in contextWindow:
                print(
                    f"\n=== Model: {model_name}, Prompt Type: {prompt_type}, Context Window: {context_window} ==="
                )

                file_name = f"{model_name.replace('/', '_')}_{context_window}.csv"
                csv_path = prompt_dir / file_name
                create_csv_with_headers(csv_path)

                for idx, row in tangled_df.iterrows():
                    shas = json.loads(row["shas"])
                    commits: Dict[str, str] = {}
                    for sha in shas:
                        matching_row = atomic_df[atomic_df["sha"] == sha]
                        commit_message = matching_row["masked_commit_message"].values[0]
                        git_diff = matching_row["git_diff"].values[0]
                        commits[sha] = {
                            "commit_message": commit_message,
                            "git_diff": git_diff,
                        }

                    system_prompt = get_prompt()
                    truncated_commit = truncate_commits(commits, context_window)
                    print(truncated_commit)
                    try:
                        prediction, inference_time = llms.api_call(
                            model_name=model_name,
                            commit=truncated_commit,
                            system_prompt=system_prompt,
                        )
                        predicted_types = llms.parse_model_response(prediction)
                    except Exception as e:
                        print(f"Error processing row {idx}: {e}")
                        predicted_types = []
                        inference_time = 0.0

                    result = {
                        "row_index": idx,
                        "context_window": context_window,
                        "predicted_types": predicted_types,
                        "actual_types": json.loads(row["types"]),
                        "inference_time": inference_time,
                        "shas": shas,
                    }

                    result_df = pd.DataFrame([result])
                    result_df.to_csv(csv_path, mode="a", header=False, index=False)

                    print(
                        f"Row {idx}: inference_time(sec): {inference_time:.2f} - saved to {csv_path}"
                    )

        llms.clear_cache()

    print(f"\nDataset Summary:")
    print(f"Loaded tangled dataset: {len(tangled_df)} samples")
    print(f"Loaded atomic dataset: {len(atomic_df)} samples")
    print(f"Processed models: {', '.join(model_names)}")
    print(f"Results saved in results/ directory, organized by prompt type")
