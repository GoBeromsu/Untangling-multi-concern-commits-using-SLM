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
    load_dataset,
    calculate_metrics,
    save_results,
    plot_graph,
    load_openai_client,
    get_openai_prediction,
    parse_prediction_to_set,
    parse_ground_truth_to_set,
)
from utils.prompt import get_system_prompt_with_message


def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Running {config['experiment_name']}")

    results = []
    accuracy_data = []

    # Use small dataset for testing
    df = load_dataset("test_small")

    # Process each context length
    for context_length in config["context_lengths"]:
        print(f"\nProcessing context length: {context_length}")

        # Use OpenAI models instead of HuggingFace for testing
        for model_type in ["slm", "llm"]:
            if model_type in config["models"]:
                for model_name in config["models"][model_type]:
                    if model_type == "llm":
                        model_name = "gpt-4-turbo"  # Use OpenAI for testing
                    else:
                        continue  # Skip SLM for now, focus on OpenAI testing

                    print(f"Processing model: {model_name}")

                    # Load OpenAI model
                    model_info = load_openai_client(model_name)

                    model_results = []
                    system_prompt = get_system_prompt_with_message()

                    # Process each sample
                    for idx, sample in df.iterrows():
                        description = sample.get("description", "")
                        diff = sample.get("diff", "")

                        # Truncate diff to simulate context length limitation
                        # Simple truncation for testing (could be more sophisticated)
                        if len(diff) > context_length * 4:  # Rough token estimation
                            diff = diff[: context_length * 4] + "..."

                        user_prompt = f"<commit_message>{description}</commit_message>\n<commit_diff>{diff}</commit_diff>"

                        try:
                            # Get prediction
                            prediction = get_openai_prediction(
                                model_info,
                                user_prompt,
                                system_prompt,
                                config["temperature"],
                                config["max_tokens"],
                            )

                            # Check if prediction is an error message
                            if (
                                prediction.startswith("An ")
                                and "error occurred" in prediction
                            ):
                                print(
                                    f"API Error for sample {idx} with {model_name}: {prediction}"
                                )
                                predicted_concerns = set()
                            else:
                                # Parse structured JSON output to extract concern types
                                predicted_concerns = parse_prediction_to_set(prediction)

                        except Exception as e:
                            print(
                                f"Error processing sample {idx} with {model_name}: {e}"
                            )
                            predicted_concerns = set()

                        # Parse ground truth concerns
                        ground_truth_concerns = parse_ground_truth_to_set(
                            sample["types"]
                        )

                        model_results.append(
                            {
                                "sample_id": idx,
                                "model": model_name,
                                "context_length": context_length,
                                "predictions": predicted_concerns,
                                "ground_truth": ground_truth_concerns,
                                "latency": 0.0,  # Not measuring latency for OpenAI testing
                                "raw_output": prediction,
                            }
                        )

                        print(
                            f"Sample {idx} with {model_name} (context {context_length})"
                        )
                        print(f"Predicted concerns: {predicted_concerns}")
                        print(f"Ground truth concerns: {ground_truth_concerns}")

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
        if len(length_df) > 0:
            metrics = calculate_metrics(length_df)
            all_metrics[f"context_{context_length}"] = metrics

    # Save results
    save_results(results_df, all_metrics, config["output_dir"])

    # Create accuracy vs length plot if we have data
    if len(accuracy_df) > 0:
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
    if len(accuracy_df) > 0:
        print(f"Plot saved to {config['output_dir']}/accuracy_vs_length.png")


if __name__ == "__main__":
    main()
