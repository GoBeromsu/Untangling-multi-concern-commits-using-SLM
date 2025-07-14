"""RQ0: Performance Gap Analysis - Compares performance of different models on commit untangling task."""

import yaml
import sys
import pandas as pd
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    load_dataset,
    calculate_metrics,
    save_metric_csvs,
)
from utils.prompt import get_system_prompt_with_message
from openai_handler import load_openai_client, get_openai_prediction
from lmstudio_handler import (
    load_lmstudio_client,
    get_lmstudio_prediction,
)


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_split = config["dataset_split"]
    slm_models = config["models"].get("slm", [])
    llm_models = config["models"].get("llm", [])
    output_dir = config["output_dir"]
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]

    df = load_dataset(dataset_split)
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
            model_info = load_lmstudio_client(model_name)

        for idx, sample in df.iterrows():
            description = sample.get("description", "")
            diff = sample.get("diff", "")
            user_prompt = f"<commit_message>{description}</commit_message>\n<commit_diff>{diff}</commit_diff>"

            # LLM API calls with error handling
            try:
                if is_llm:
                    prediction = get_openai_prediction(
                        model_info, user_prompt, system_prompt, temperature, max_tokens
                    )
                else:
                    prediction = get_lmstudio_prediction(
                        model_info, user_prompt, system_prompt, temperature, max_tokens
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
                print(
                    f"JSON decode error processing sample {idx} with {model_name}: {e}"
                )
                print(f"Raw prediction: {prediction}")
                predicted_concerns = set()
            except Exception as e:
                print(f"Error processing sample {idx} with {model_name}: {e}")
                predicted_concerns = set()

            # Parse ground truth concerns
            ground_truth_concerns = set(json.loads(sample["types"]))

            # Save result immediately after each API response
            results.append(
                {
                    "model": model_name,
                    "predictions": predicted_concerns,
                    "ground_truth": ground_truth_concerns,
                }
            )
            print(f"Sample {idx} with {model_name}")
            print(f"Predicted concerns: {predicted_concerns}")
            print(f"Ground truth concerns: {ground_truth_concerns}")

            # Calculate and save metrics in real-time with accumulated results
            results_df = pd.DataFrame(results)
            model_df = results_df[results_df["model"] == model_name]

            if len(model_df) > 0:
                metrics = calculate_metrics(model_df)
                save_metric_csvs(model_name, metrics, output_dir)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples for {model_name}")
                print(
                    f"Current F1: {metrics['f1_score']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}"
                )

        print(f"Completed {model_name}")

    # Save detailed results for debugging
    results_df = pd.DataFrame(results)
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
