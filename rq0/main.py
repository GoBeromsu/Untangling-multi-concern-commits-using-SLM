"""RQ0: Performance Gap Analysis - Compares performance of different models on commit untangling task."""

import yaml
import sys
import pandas as pd
import json
from pathlib import Path
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    load_model_and_tokenizer,
    get_prediction,
    parse_model_output,
    calculate_metrics,
    save_metric_csvs,
)
from utils.prompt import get_system_prompt_with_message
from openai_handler import load_openai_client, get_openai_prediction


def create_user_prompt(description: str, diff: str) -> str:
    """Create user prompt from commit message and diff."""
    return f"<commit_message>{description}</commit_message>\n<commit_diff>{diff}</commit_diff>"


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset_name"]
    dataset_split = config["dataset_split"]
    slm_models = config["models"].get("slm", [])
    llm_models = config["models"].get("llm", [])
    output_dir = config["output_dir"]
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]

    df = load_dataset(dataset_name, dataset_split)
    system_prompt = get_system_prompt_with_message()
    results = []

    # Process all models
    all_models = [(model, False) for model in slm_models] + [
        (model, True) for model in llm_models
    ]

    for model_name, is_llm in all_models:
        model_type = "LLM" if is_llm else "SLM"
        print(f"Processing {model_type} model: {model_name}")

        # Load model based on type
        if is_llm:
            model_info = load_openai_client(model_name)
        else:
            model_info = load_model_and_tokenizer(model_name)

        for idx, sample in df.iterrows():
            # Create user prompt from description and diff
            description = sample.get("description", "")
            diff = sample.get("diff", "")
            user_prompt = create_user_prompt(description, diff)

            if is_llm:
                prediction = get_openai_prediction(
                    model_info, user_prompt, system_prompt, temperature, max_tokens
                )
            else:
                # For SLM, combine system and user prompts
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                prediction, _ = get_prediction(
                    model_info, full_prompt, temperature, max_tokens
                )

            predicted_concerns = parse_model_output(prediction)

            # Extract ground truth from types column (assuming it's a JSON string or list)
            ground_truth_raw = sample.get("types", "[]")
            if isinstance(ground_truth_raw, str):
                try:
                    ground_truth_concerns = set(json.loads(ground_truth_raw))
                except:
                    ground_truth_concerns = set()
            elif isinstance(ground_truth_raw, list):
                ground_truth_concerns = set(ground_truth_raw)
            else:
                ground_truth_concerns = set()

            results.append(
                {
                    "sample_id": idx,
                    "model": model_name,
                    "predictions": predicted_concerns,
                    "ground_truth": ground_truth_concerns,
                    "raw_output": prediction,
                }
            )

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples")

    results_df = pd.DataFrame(results)

    # Calculate metrics for each model
    all_metrics = {}
    all_model_names = slm_models + llm_models

    for model_name in all_model_names:
        model_df = results_df[results_df["model"] == model_name]
        metrics = calculate_metrics(model_df)
        all_metrics[model_name] = metrics
        print(
            f"{model_name} - F1: {metrics['f1_score']:.3f}, "
            f"Precision: {metrics['precision']:.3f}, "
            f"Recall: {metrics['recall']:.3f}"
        )

    # Save results as 3 separate CSV files
    save_metric_csvs(all_metrics, output_dir)

    # Also save detailed results for debugging
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path / "detailed_predictions.csv", index=False)

    print(f"Results saved to {output_dir}")
    print("Files created:")
    print("- macro_f1.csv")
    print("- macro_precision.csv")
    print("- macro_recall.csv")
    print("- detailed_predictions.csv")


if __name__ == "__main__":
    main()
