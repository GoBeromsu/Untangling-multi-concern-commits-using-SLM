"""Data utilities for loading datasets and creating prompts."""

from typing import Dict, List, Any, Optional
from datasets import load_dataset
import pandas as pd


def load_dataset(
    name: str, split: str, config_name: Optional[str] = None
) -> pd.DataFrame:
    """Load dataset from Hugging Face Hub."""
    dataset = load_dataset(name, config_name, split=split)
    return pd.DataFrame(dataset)


def create_prompt(
    sample: Dict[str, Any],
    template: str,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    with_message: bool = True,
) -> str:
    """Create final prompt for model inference."""
    # Build few-shot examples if provided
    few_shot_text = ""
    if few_shot_examples:
        for example in few_shot_examples:
            few_shot_text += f"Example:\n{example}\n\n"

    # Prepare sample content
    content = f"Commit Diff:\n{sample['diff']}\n"
    if with_message and "message" in sample:
        content += f"Commit Message: {sample['message']}\n"

    return template.format(few_shot_examples=few_shot_text, content=content)
