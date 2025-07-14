"""Model utilities for loading HuggingFace models and generating predictions."""

from typing import Dict, Any, Tuple
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_id: str) -> Dict[str, Any]:
    """Load HuggingFace model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )

    return {
        "type": "huggingface",
        "model": model,
        "tokenizer": tokenizer,
        "model_name": model_id,
    }


def get_prediction(
    model_info: Dict[str, Any], prompt: str, temperature: float, max_tokens: int
) -> Tuple[str, float]:
    """Get prediction from HuggingFace model with latency measurement."""
    start_time = time.time()

    model = model_info["model"]
    tokenizer = model_info["tokenizer"]

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
        )

    generated_tokens = outputs[0][inputs.shape[1] :]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return prediction, time.time() - start_time
