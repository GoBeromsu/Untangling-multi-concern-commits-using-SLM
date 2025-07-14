"""Model utilities for loading models and generating predictions."""

from typing import Dict, Any, Tuple
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from dotenv import load_dotenv
import os

load_dotenv()


def load_model_and_tokenizer(model_id: str) -> Dict[str, Any]:
    """Load model and tokenizer."""
    if "gpt-4" in model_id.lower():
        return {
            "type": "openai",
            "model_name": model_id,
            "client": openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        }

    # HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return {
        "type": "huggingface",
        "model": model,
        "tokenizer": tokenizer,
        "model_name": model_id,
    }


def get_prediction(model_info: Dict[str, Any], prompt: str) -> Tuple[str, float]:
    """Get prediction with latency measurement."""
    start_time = time.time()

    if model_info["type"] == "openai":
        response = model_info["client"].chat.completions.create(
            model=model_info["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
        )
        prediction = response.choices[0].message.content
    else:
        # HuggingFace inference
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]

        inputs = tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=1000,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )

        generated_tokens = outputs[0][inputs.shape[1] :]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return prediction, time.time() - start_time
