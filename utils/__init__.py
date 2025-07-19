"""
Utility modules for commit untangler experiments.
"""

from .model import load_model_and_tokenizer, get_prediction
from .llms import (
    openai_api_call,
    load_openai_client,
    get_openai_prediction,
    get_models,
    load_model,
    api_call,
    clear_cache,
    parse_model_response,
    parse_prediction_to_set,
    parse_ground_truth_to_set,
)
from .eval import (
    load_dataset,
    get_tp_fp_fn,
    calculate_metrics,
    calculate_batch_metrics,
    save_results,
    save_metric_csvs,
    plot_graph,
)

__all__ = [
    "load_model_and_tokenizer",
    "get_prediction",
    # Unified LLM interfaces
    "openai_api_call",
    "load_openai_client",
    "get_openai_prediction",
    "get_models",
    "load_model",
    "api_call",
    "clear_cache",
    "parse_model_response",
    "parse_prediction_to_set",
    "parse_ground_truth_to_set",
    # Evaluation utilities
    "load_dataset",
    "get_tp_fp_fn",
    "calculate_metrics",
    "calculate_batch_metrics",
    "save_results",
    "save_metric_csvs",
    "plot_graph",
]
