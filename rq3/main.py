"""RQ3: Latency vs. Context Length - Analyzes the relationship between context length and model inference latency."""

import yaml
import sys
import pandas as pd
import json
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    load_dataset,
    calculate_metrics,
    save_results,
    plot_graph,
    load_openai_client,
    get_openai_prediction,
)
from utils.prompt import get_system_prompt_with_message


def get_openai_prediction_with_latency(
    model_info, user_prompt, system_prompt, temperature, max_tokens
):
    """Get OpenAI prediction with latency measurement."""
    start_time = time.time()

    # Use the existing get_openai_prediction function
    from utils.openai import get_openai_prediction as base_get_prediction

    prediction = base_get_prediction(
        model_info, user_prompt, system_prompt, temperature, max_tokens
    )

    latency = time.time() - start_time
    return prediction, latency


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Running {config['experiment_name']}")

    results = []
    latency_data = []

    # Use small dataset for testing
    df = load_dataset("test_small")

    for context_length in config["context_lengths"]:
        print(f"\nProcessing context length: {context_length}")

        # Use OpenAI model instead of HuggingFace for testing
        model_name = "gpt-4-turbo"  # Use OpenAI for testing
        print(f"Processing model: {model_name}")

        model_info = load_openai_client(model_name)
        model_results = []
        latencies = []
        system_prompt = get_system_prompt_with_message()

        for idx, sample in df.iterrows():
            description = sample.get("description", "")
            diff = sample.get("diff", "")

            # Truncate diff to simulate context length limitation
            # Simple truncation for testing (could be more sophisticated)
            if len(diff) > context_length * 4:  # Rough token estimation
                diff = diff[: context_length * 4] + "..."

            user_prompt = f"<commit_message>{description}</commit_message>\n<commit_diff>{diff}</commit_diff>"

            try:
                prediction, latency = get_openai_prediction_with_latency(
                    model_info,
                    user_prompt,
                    system_prompt,
                    config["temperature"],
                    config["max_tokens"],
                )
                latencies.append(latency)

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
                latencies.append(0.0)  # Add placeholder latency
            except Exception as e:
                print(f"Error processing sample {idx} with {model_name}: {e}")
                predicted_concerns = set()
                latencies.append(0.0)  # Add placeholder latency

            # Parse ground truth concerns
            ground_truth_concerns = set(json.loads(sample["types"]))

            model_results.append(
                {
                    "sample_id": idx,
                    "model": model_name,
                    "context_length": context_length,
                    "predictions": predicted_concerns,
                    "ground_truth": ground_truth_concerns,
                    "latency": latency if "latency" in locals() else 0.0,
                    "raw_output": prediction,
                }
            )

            print(
                f"Sample {idx} with {model_name} (context {context_length}) - Latency: {latency:.3f}s"
            )
            print(f"Predicted concerns: {predicted_concerns}")
            print(f"Ground truth concerns: {ground_truth_concerns}")

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples")

        # Calculate latency statistics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            latency_data.append(
                {
                    "context_length": context_length,
                    "model": model_name,
                    "avg_latency": avg_latency,
                    "min_latency": min_latency,
                    "max_latency": max_latency,
                }
            )

            print(
                f"Context {context_length} - Avg Latency: {avg_latency:.3f}s, "
                f"Min: {min_latency:.3f}s, Max: {max_latency:.3f}s"
            )

        results.extend(model_results)

    results_df = pd.DataFrame(results)
    latency_df = pd.DataFrame(latency_data)

    # Calculate latency metrics for each context length
    latency_metrics = {}
    for context_length in config["context_lengths"]:
        length_df = results_df[results_df["context_length"] == context_length]
        if len(length_df) > 0:
            latency_metrics[f"context_{context_length}"] = {
                "avg_latency": length_df["latency"].mean(),
                "std_latency": length_df["latency"].std(),
                "min_latency": length_df["latency"].min(),
                "max_latency": length_df["latency"].max(),
            }

    save_results(results_df, latency_metrics, config["output_dir"])

    # Create latency vs length plot if we have data
    if len(latency_df) > 0:
        plot_graph(
            latency_df,
            "context_length",
            "avg_latency",
            f"{config['output_dir']}/latency_vs_length.png",
            title="Latency vs Context Length",
            xlabel="Context Length (tokens)",
            ylabel="Average Latency (seconds)",
        )

    print(f"Results saved to {config['output_dir']}")
    if len(latency_df) > 0:
        print(f"Plot saved to {config['output_dir']}/latency_vs_length.png")


if __name__ == "__main__":
    main()
