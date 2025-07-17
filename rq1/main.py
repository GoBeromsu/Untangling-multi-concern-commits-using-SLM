"""RQ1: Context Ablation Study - Studies the impact of commit message context on model performance."""

import yaml
import sys
import pandas as pd
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    load_dataset,
    calculate_metrics,
    save_results,
    load_openai_client,
    get_openai_prediction,
)
from utils.prompt import get_system_prompt_diff_only


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Running {config['experiment_name']}")

    # Use small dataset for testing
    df = load_dataset("test_small")
    results = []

    # Use OpenAI model instead of local models for testing
    model_name = "gpt-4-turbo"  # Use OpenAI model
    print(f"Processing model: {model_name}")
    print(f"Include message: {config['include_message']}")

    model_info = load_openai_client(model_name)
    model_results = []

    # Get appropriate system prompt based on include_message setting
    if config["include_message"]:
        from utils.prompt import get_system_prompt_with_message

        system_prompt = get_system_prompt_with_message()
    else:
        system_prompt = get_system_prompt_diff_only()

    for idx, sample in df.iterrows():
        description = sample.get("description", "")
        diff = sample.get("diff", "")

        # Create user prompt based on include_message setting
        if config["include_message"]:
            user_prompt = f"<commit_message>{description}</commit_message>\n<commit_diff>{diff}</commit_diff>"
        else:
            user_prompt = f"<commit_diff>{diff}</commit_diff>"

        try:
            prediction = get_openai_prediction(
                model_info,
                user_prompt,
                system_prompt,
                config["temperature"],
                config["max_tokens"],
            )

            # Check if prediction is an error message
            if prediction.startswith("An ") and "error occurred" in prediction:
                print(f"API Error for sample {idx} with {model_name}: {prediction}")
                predicted_concerns = set()
            else:
                # Parse structured JSON output to extract concern types
                output_json = json.loads(prediction)
                predicted_concerns = set(output_json["types"])

        except json.JSONDecodeError as e:
            print(f"JSON decode error processing sample {idx} with {model_name}: {e}")
            print(f"Raw prediction: {prediction}")
            predicted_concerns = set()
        except Exception as e:
            print(f"Error processing sample {idx} with {model_name}: {e}")
            predicted_concerns = set()

        # Parse ground truth concerns
        ground_truth_concerns = set(json.loads(sample["types"]))

        model_results.append(
            {
                "sample_id": idx,
                "model": model_name,
                "predictions": predicted_concerns,
                "ground_truth": ground_truth_concerns,
                "raw_output": prediction,
            }
        )

        print(f"Sample {idx} with {model_name}")
        print(f"Predicted concerns: {predicted_concerns}")
        print(f"Ground truth concerns: {ground_truth_concerns}")

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples")

    results.extend(model_results)

    results_df = pd.DataFrame(results)

    # Calculate metrics for the model
    all_metrics = {}
    model_df = results_df[results_df["model"] == model_name]
    metrics = calculate_metrics(model_df)
    all_metrics[model_name] = metrics
    print(
        f"{model_name} ({'with' if config['include_message'] else 'without'} message) - F1: {metrics['f1_score']:.3f}, "
        f"Precision: {metrics['precision']:.3f}, "
        f"Recall: {metrics['recall']:.3f}"
    )

    save_results(results_df, all_metrics, config["output_dir"])
    print(f"Results saved to {config['output_dir']}")


if __name__ == "__main__":
    main()
