"""
RQ2: Accuracy vs. Context Length
Analyzes the relationship between context length and model accuracy.
"""

import yaml
import sys
import pandas as pd
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    load_dataset_from_hf,
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
    accuracy_data = []

    # Process each context length
    for context_length in config["context_lengths"]:
        print(f"\nProcessing context length: {context_length}")

        # Load dataset with specific context length configuration
        df = load_dataset_from_hf(
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

            # Process each sample
            for idx, sample in df.iterrows():
                # Create prompt
                prompt = create_prompt(
                    sample.to_dict(),
                    config["prompt_template"],
                    with_message=config["include_message"],
                )

                # Get prediction
                prediction, latency = get_prediction(model_info, prompt)

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

            # Calculate metrics for this context length and model
            model_df = pd.DataFrame(model_results)
            metrics = calculate_metrics(model_df)

            # Store accuracy data for plotting
            accuracy_data.append(
                {
                    "context_length": context_length,
                    "model": model_name,
                    "f1_score": metrics["f1_score"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                }
            )

            print(
                f"Context {context_length} - F1: {metrics['f1_score']:.3f}, "
                f"Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}"
            )

            results.extend(model_results)

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    accuracy_df = pd.DataFrame(accuracy_data)

    # Calculate overall metrics
    all_metrics = {}
    for context_length in config["context_lengths"]:
        length_df = results_df[results_df["context_length"] == context_length]
        metrics = calculate_metrics(length_df)
        all_metrics[f"context_{context_length}"] = metrics

    # Save results
    save_results(results_df, all_metrics, config["output_dir"])

    # Create accuracy vs length plot
    plot_graph(
        accuracy_df,
        "context_length",
        "f1_score",
        f"{config['output_dir']}/accuracy_vs_length.png",
        title="Accuracy vs Context Length",
        xlabel="Context Length (tokens)",
        ylabel="F1 Score",
    )

    print(f"Results saved to {config['output_dir']}")
    print(f"Plot saved to {config['output_dir']}/accuracy_vs_length.png")


if __name__ == "__main__":
    main()
