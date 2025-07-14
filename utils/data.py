"""
Data utilities for loading datasets and creating prompts.
"""

from typing import Dict, List, Any, Optional
from datasets import load_dataset
import pandas as pd


def load_dataset_from_hf(
    name: str, split: str, config_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Load dataset from Hugging Face Hub.

    Args:
        name: Dataset name on Hugging Face Hub
        split: Dataset split ('train', 'test', 'validation')
        config_name: Optional config name for different dataset configurations

    Returns:
        DataFrame containing the loaded dataset
    """
    dataset = load_dataset(name, config_name, split=split)
    return pd.DataFrame(dataset)


def create_prompt(
    sample: Dict[str, Any],
    template: str,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    with_message: bool = True,
) -> str:
    """
    Create final prompt for model inference.

    Args:
        sample: Single data sample containing diff and message
        template: Prompt template string
        few_shot_examples: Optional list of few-shot examples
        with_message: Whether to include commit message in prompt

    Returns:
        Formatted prompt string
    """
    # Build few-shot examples if provided
    few_shot_text = ""
    if few_shot_examples:
        for example in few_shot_examples:
            few_shot_text += f"Example:\n{example}\n\n"

    # Prepare sample content
    content = f"Commit Diff:\n{sample['diff']}\n"
    if with_message and "message" in sample:
        content += f"Commit Message: {sample['message']}\n"

    # Format final prompt
    prompt = template.format(few_shot_examples=few_shot_text, content=content)

    return prompt
