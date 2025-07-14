"""RQ0: Performance Gap Analysis - Compares performance of different models on commit untangling task."""

import yaml
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple

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
from utils.prompt import get_system_prompt_with_message
from openai_handler import load_openai_client, get_openai_prediction


def get_model_handlers(is_llm: bool) -> Tuple[Callable, Callable]:
    """Get both model loader and prediction functions based on model type.

    Returns:
        Tuple[Callable, Callable]: (model_loader_func, prediction_func)
    """
    if is_llm:
        return load_openai_client, get_openai_prediction
    else:
        return load_model_and_tokenizer, get_prediction


def process_sample(
    sample: pd.Series,
    include_message: bool,
    model_info: Any,
    model_name: str,
    temperature: float,
    max_tokens: int,
    idx: int,
    prediction_func: Callable,
) -> Dict[str, Any]:
    """Process a single sample with the given model."""
    prompt_template = get_system_prompt_with_message()
    prompt = create_prompt(
        sample.to_dict(),
        prompt_template,
        with_message=include_message,
    )

    prediction = prediction_func(model_info, prompt, temperature, max_tokens)
    predicted_concerns = parse_model_output(prediction)
    ground_truth_concerns = set(sample.get("concerns", []))

    return {
        "sample_id": idx,
        "model": model_name,
        "predictions": predicted_concerns,
        "ground_truth": ground_truth_concerns,
        "raw_output": prediction,
    }


def process_model_type(
    model_names: List[str],
    df: pd.DataFrame,
    config: Dict[str, Any],
    is_llm: bool = False,
) -> List[Dict[str, Any]]:
    """Process all models of a specific type (SLM or LLM)."""
    results = []
    model_loader_func, prediction_func = get_model_handlers(is_llm)

    for model_name in model_names:
        print(f"Processing model: {model_name}")
        model_info = model_loader_func(model_name)
        model_results = []

        for idx, sample in df.iterrows():
            result = process_sample(
                sample=sample,
                include_message=config["include_message"],
                model_info=model_info,
                model_name=model_name,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                idx=idx,
                prediction_func=prediction_func,
            )
            model_results.append(result)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples")

        results.extend(model_results)

    return results


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Running {config['experiment_name']}")

    df = load_dataset(config["dataset_name"], config["dataset_split"])
    results = []

    # Process SLM models
    if "slm" in config["models"]:
        slm_results = process_model_type(
            model_names=config["models"]["slm"], df=df, config=config, is_llm=False
        )
        results.extend(slm_results)

    # Process LLM models
    if "llm" in config["models"]:
        llm_results = process_model_type(
            model_names=config["models"]["llm"], df=df, config=config, is_llm=True
        )
        results.extend(llm_results)

    results_df = pd.DataFrame(results)

    # Calculate metrics for each model
    all_metrics = {}
    all_model_names = config["models"].get("slm", []) + config["models"].get("llm", [])

    for model_name in all_model_names:
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
