"""
RQ3: Latency vs. Context Length
Analyzes the relationship between context length and model inference latency.
"""

import yaml
import sys
import pandas as pd
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    load_dataset,
    create_prompt,
    load_model_and_tokenizer,
    get_prediction,
    parse_model_output,
    calculate_metrics,
    save_results,
    plot_graph,
)


def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Running {config['experiment_name']}")

    results = []
    latency_data = []

    # Process each context length
    for context_length in config["context_lengths"]:
        print(f"\nProcessing context length: {context_length}")

        # Load dataset with specific context length configuration
        df = load_dataset(
            config["dataset_name"],
            config["dataset_split"],
            config_name=str(context_length),  # Use context length as config name
        )

        # Process each model
        for model_name in config["models"]:
            print(f"Processing model: {model_name}")

            # Load model
            model_info = load_model_and_tokenizer(model_name)

            model_results = []
            latencies = []

            # Process each sample
            for idx, sample in df.iterrows():
                # Create prompt
                prompt = create_prompt(
                    sample.to_dict(),
                    config["prompt_template"],
                    with_message=config["include_message"],
                )

                # Get prediction with latency measurement
                prediction, latency = get_prediction(model_info, prompt)
                latencies.append(latency)

                # Parse output
                predicted_concerns = parse_model_output(prediction)
                ground_truth_concerns = set(sample.get("concerns", []))

                model_results.append(
                    {
                        "sample_id": idx,
                        "model": model_name,
                        "context_length": context_length,
                        "predictions": predicted_concerns,
                        "ground_truth": ground_truth_concerns,
                        "latency": latency,
                        "raw_output": prediction,
                    }
                )

                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(df)} samples")

            # Calculate latency statistics
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            # Store latency data for plotting
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

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    latency_df = pd.DataFrame(latency_data)

    # Calculate latency metrics for each context length
    latency_metrics = {}
    for context_length in config["context_lengths"]:
        length_df = results_df[results_df["context_length"] == context_length]
        latency_metrics[f"context_{context_length}"] = {
            "avg_latency": length_df["latency"].mean(),
            "std_latency": length_df["latency"].std(),
            "min_latency": length_df["latency"].min(),
            "max_latency": length_df["latency"].max(),
        }

    # Save results
    save_results(results_df, latency_metrics, config["output_dir"])

    # Create latency vs length plot
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
    print(f"Plot saved to {config['output_dir']}/latency_vs_length.png")


if __name__ == "__main__":
    main()
