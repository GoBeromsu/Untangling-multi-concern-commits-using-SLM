import sys
import logging
from typing import Dict, Any

import datasets
from datasets import load_dataset
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)

"""
Multi-Concern Commit Classification SFT Training Script
University of Sheffield HPC Stanage - A100 GPU Setup

This script fine-tunes a language model to classify multi-concern commits using SFTTrainer.
Dataset: Berom0227/Untangling-Multi-Concern-Commits-with-Small-Language-Models
Input: description (commit message), types, reason
Target: Learning to predict reasoning for commit classification

HPC Setup Requirements:
- GPU: A100 80GB, 4x per node, requires SLURM job submission
- Memory: 256GB RAM per node, use --mem flag for allocation
- Modules: module load PyTorch/1.13.1-foss-2022a-CUDA-11.7.0

Setup Steps:
1. Install dependencies:
    conda install -c conda-forge accelerate=1.3.0
    pip3 install -i https://pypi.org/simple/ bitsandbytes
    pip3 install peft==0.14.0
    pip3 install transformers==4.48.1
    pip3 install trl datasets
    pip3 install deepspeed

2. Setup accelerate config for A100:
    accelerate config

3. Run with accelerate:
    accelerate launch train.py
"""

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a software engineer classifying individual code units extracted from a tangled commit.
Each change unit (e.g., function, method, class, or code block) represents a reviewable atomic change, and must be assigned exactly one label.

Label selection must assign exactly one concern from the following unified set:
- Purpose labels : the motivation behind making a code change (feat, fix, refactor)
- Object labels : the essence of the code changes that have been made(docs, test, cicd, build)
     - Use an object label only when the code unit is fully dedicated to that artifact category (e.g., writing test logic, modifying documentation).

# Instructions
1. Review the code unit and determine the most appropriate label from the unified set.
2. If multiple labels seem possible, resolve the overlap by applying the following rule:
     - **Purpose + Purpose**: Choose the label that best reflects *why* the change was made — `fix` if resolving a bug, `feat` if adding new capability, `refactor` if improving structure without changing behavior.
     - **Object + Object**: Choose the label that reflects the *functional role* of the artifact being modified — e.g., even if changing build logic, editing a CI script should be labeled as `cicd`.
     - **Purpose + Object**: If the change is driven by code behavior (e.g., fixing test logic), assign a purpose label; if it is entirely scoped to a support artifact (e.g., adding new tests), assign an object label.

# Labels
- feat: Introduces new features to the codebase.
- fix: Fixes bugs or faults in the codebase.
- refactor: Restructures existing code without changing external behavior (e.g., improves readability, simplifies complexity, removes unused code).
- docs: Modifies documentation or text (e.g., fixes typos, updates comments or docs).
- test: Modifies test files (e.g., adds or updates tests).
- cicd: Updates CI (Continuous Integration) configuration files or scripts (e.g., `.travis.yml`, `.github/workflows`).
- build: Affects the build system (e.g., updates dependencies, changes build configs or scripts).# Example
"""

# Training constants
DEFAULT_BATCH_SIZE: int = 4
NUM_WORKERS: int = 4
MAX_SEQ_LENGTH: int = 2048
LORA_RANK: int = 16
LORA_ALPHA: int = 32
LEARNING_RATE: float = 5.0e-06

###################
# Hyper-parameters
###################
training_config = {
    "bf16": True,
    "do_eval": False,
    "learning_rate": LEARNING_RATE,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": DEFAULT_BATCH_SIZE,
    "per_device_train_batch_size": DEFAULT_BATCH_SIZE,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
}

peft_config = {
    "r": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
}

train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")
logger.info(f"PEFT parameters {peft_conf}")

################
# Model Loading
################
checkpoint_path = "microsoft/Phi-4-mini-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # A100 optimized flash attention
    torch_dtype=torch.bfloat16,
    device_map=None,
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = MAX_SEQ_LENGTH
tokenizer.pad_token = (
    tokenizer.unk_token
)  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "right"


##################
# Data Processing
##################
def apply_chat_template(example, tokenizer) -> Dict[str, Any]:
    """Apply chat template for multi-concern commit classification."""
    # Create structured prompt for commit analysis
    user_content = f"""Analyze this commit for multiple concerns:

Commit Message: {example['description']}
Types: {example['types']}

Provide reasoning for the classification:"""

    assistant_content = example["reason"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example


def load_commit_dataset(repo_name: str, split: str = "train"):
    """Load multi-concern commit dataset from HuggingFace."""
    return load_dataset(repo_name, split=split)


# Load dataset
train_dataset = load_commit_dataset(
    "Berom0227/Untangling-Multi-Concern-Commits-with-Small-Language-Models", "train"
)

# For evaluation, we can use a portion of training data or create a validation split
eval_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = eval_dataset["train"]
test_dataset = eval_dataset["test"]

column_names = list(train_dataset.features)

processed_train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=NUM_WORKERS,
    remove_columns=column_names,
    desc="Applying chat template to multi-concern commit train data",
)

processed_test_dataset = test_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=NUM_WORKERS,
    remove_columns=column_names,
    desc="Applying chat template to multi-concern commit test data",
)

###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True,
)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

#############
# Evaluation
#############
tokenizer.padding_side = "left"
metrics = trainer.evaluate()
metrics["eval_samples"] = len(processed_test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

############
# Save model
############
trainer.save_model(train_conf.output_dir)
