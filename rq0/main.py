"""RQ0: Performance Gap Analysis - Compares performance of different models on commit untangling task."""

import yaml
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    load_dataset,
    create_prompt,
    load_model_and_tokenizer,
    get_prediction,
    parse_model_output,
    calculate_metrics,
    save_results,
)
from openai_handler import load_openai_client, get_openai_prediction


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Running {config['experiment_name']}")

    df = load_dataset(config["dataset_name"], config["dataset_split"])
    results = []

    for model_name in config["models"]:
        print(f"Processing model: {model_name}")

        # Use different handlers for different model types
        if "gpt-4" in model_name.lower():
            model_info = load_openai_client(model_name)
        else:
            model_info = load_model_and_tokenizer(model_name)

        model_results = []

        for idx, sample in df.iterrows():
            prompt = create_prompt(
                sample.to_dict(),
                config["prompt_template"],
                with_message=config["include_message"],
            )

            # Use appropriate prediction function based on model type
            if "gpt-4" in model_name.lower():
                prediction, latency = get_openai_prediction(
                    model_info, prompt, config["temperature"], config["max_tokens"]
                )
            else:
                prediction, latency = get_prediction(
                    model_info, prompt, config["temperature"], config["max_tokens"]
                )

            predicted_concerns = parse_model_output(prediction)
            ground_truth_concerns = set(sample.get("concerns", []))

            model_results.append(
                {
                    "sample_id": idx,
                    "model": model_name,
                    "predictions": predicted_concerns,
                    "ground_truth": ground_truth_concerns,
                    "latency": latency,
                    "raw_output": prediction,
                }
            )

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples")

        results.extend(model_results)

    results_df = pd.DataFrame(results)

    # Calculate metrics for each model
    all_metrics = {}
    for model_name in config["models"]:
        model_df = results_df[results_df["model"] == model_name]
        metrics = calculate_metrics(model_df)
        all_metrics[model_name] = metrics
        print(
            f"{model_name} - F1: {metrics['f1_score']:.3f}, "
            f"Precision: {metrics['precision']:.3f}, "
            f"Recall: {metrics['recall']:.3f}"
        )

    save_results(results_df, all_metrics, config["output_dir"])
    print(f"Results saved to {config['output_dir']}")


if __name__ == "__main__":
    main()
