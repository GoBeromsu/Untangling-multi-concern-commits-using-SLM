# Reference : https://github.com/microsoft/PhiCookBook/blob/main/code/03.Finetuning/Phi-3-finetune-lora-python.ipynb
import sys
import logging
from typing import Dict, Any

import torch

# 'load_dataset' is a function from the 'datasets' library by Hugging Face which allows you to load a dataset.
from datasets import load_dataset

# 'LoraConfig' and 'prepare_model_for_kbit_training' are from the 'peft' library.
# 'LoraConfig' is used to configure the LoRA (Learning from Random Architecture) model.
# 'prepare_model_for_kbit_training' is a function that prepares a model for k-bit training.
# 'TaskType' contains differenct types of tasks supported by PEFT
# 'PeftModel' base model class for specifying the base Transformer model and configuration to apply a PEFT method to.
from peft import LoraConfig, TaskType

# Several classes and functions are imported from the 'transformers' library by Hugging Face.
# 'AutoModelForCausalLM' is a class that provides a generic transformer model for causal language modeling.
# 'AutoTokenizer' is a class that provides a generic tokenizer class.
# 'BitsAndBytesConfig' is a class for configuring the Bits and Bytes optimizer.
# 'TrainingArguments' is a class that defines the arguments used for training a model.
# 'set_seed' is a function that sets the seed for generating random numbers.
# 'pipeline' is a function that creates a pipeline that can process data and make predictions.
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

# 'SFTTrainer' is a class from the 'trl' library that provides a trainer for soft fine-tuning.
from trl import SFTTrainer

from utils.prompt import get_system_prompt

"""
Multi-Concern Commit Classification SFT Training Script
University of Sheffield HPC Stanage - A100 GPU Setup

This script fine-tunes a language model to classify multi-concern commits using SFTTrainer.
Dataset: Berom0227/Untangling-Multi-Concern-Commits-with-Small-Language-Models
Input: commit_message (commit message), types, reason
Target: Learning to predict reasoning for commit classification

HPC Stanage Setup Requirements:
- GPU: A100 40/80GB, requires SLURM job submission
- Memory: 128GB RAM recommended for single A100
- Modules: module load CUDA/12.4.0 && module load Anaconda3/2022.05

Setup Steps:
1. Create conda environment:
    conda env create -f fine_tuning/environment.yml
    conda activate phi4_env

2. Verify installation:
    python -c "import torch; print(torch.cuda.is_available())"

3. Run with SLURM:
    sbatch fine_tuning/prepare.sh

"""

logger = logging.getLogger(__name__)

# Model and dataset configuration
MODEL_ID: str = "microsoft/phi-4"
MODEL_NAME: str = "microsoft/phi-4"
DATASET_NAME: str = (
    "Berom0227/Untangling-Multi-Concern-Commits-with-Small-Language-Models"
)

NEW_MODEL: str = "Untangling-Multi-Concern-Commits-with-Small-Language-Models"
HF_MODEL_REPO: str = "Berom0227/" + NEW_MODEL

DEVICE_MAP: str = "auto"

# 'lora_r' is the dimension of the LoRA attention.
LORA_RANK: int = 16

# 'lora_alpha' is the alpha parameter for LoRA scaling.
LORA_ALPHA: int = 16

# 'lora_dropout' is the dropout probability for LoRA layers.
LORA_DROPOUT: float = 0.05

# 'target_modules' is a list of the modules in the model that will be replaced with LoRA layers.
TARGET_MODULES: list[str] = [
    "k_proj",
    "q_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "down_proj",
    "up_proj",
]

# Training configuration
MAX_SEQ_LENGTH: int = 16_384
NUM_WORKERS: int = 4

set_seed(1234)

## Dataset Loading
train_dataset = load_dataset(
    DATASET_NAME,
    split="train",
)

test_dataset = load_dataset(
    DATASET_NAME,
    split="test",
)

# Load tokenizer to prepare the dataset (First tokenizer - for data formatting only)
tokenizer_id = MODEL_ID

# 'AutoTokenizer.from_pretrained' is a method that loads a tokenizer from the Hugging Face Model Hub.
# 'tokenizer_id' is passed as an argument to specify which tokenizer to load.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

# 'tokenizer.padding_side' is a property that specifies which side to pad when the input sequence is shorter than the maximum sequence length.
# Setting it to 'right' means that padding tokens will be added to the right (end) of the sequence.
# This is done to prevent warnings that can occur when the padding side is not explicitly set.
tokenizer.padding_side = "right"

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


##################
# Data Processing
##################
def create_message_column(row) -> Dict[str, Any]:
    """Create messages column for multi-concern commit classification."""
    # Create structured prompt for commit analysis
    user_content = f"# Commit Message\n{row['commit_message']}\n\n# Diff\n```diff\n{row['diff']}\n```\n"
    assistant_content = (
        f"# Reasoning\n{row['reason']}\n# Result types\n{row['types']}\n"
    )

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return {"messages": messages}


# 'format_dataset_chatml' is a function that takes a row from the dataset and returns a dictionary
# with a 'text' key and a string of formatted chat messages as its value.
# Uses the first tokenizer with tokenize=False to create formatted strings (not token IDs)
def format_dataset_chatml(row) -> Dict[str, Any]:
    """Format dataset with chat template for multi-concern commit classification."""
    return {
        "text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
    }


column_names = list(train_dataset.features)

# Step 1: Create messages column
train_dataset_with_messages = train_dataset.map(
    create_message_column,
    num_proc=NUM_WORKERS,
    desc="Creating messages column for multi-concern commit train data",
)

# Step 2: Format with chat template
processed_train_dataset = train_dataset_with_messages.map(
    format_dataset_chatml,
    num_proc=NUM_WORKERS,
    remove_columns=column_names,
    desc="Applying chat template to multi-concern commit train data",
)

# Process test dataset
test_dataset_with_messages = test_dataset.map(
    create_message_column,
    num_proc=NUM_WORKERS,
    desc="Creating messages column for multi-concern commit test data",
)

processed_test_dataset = test_dataset_with_messages.map(
    format_dataset_chatml,
    num_proc=NUM_WORKERS,
    remove_columns=column_names,
    desc="Applying chat template to multi-concern commit test data",
)

###########
# Training
###########

if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    compute_dtype = torch.float16
    attn_implementation = "sdpa"


# Second tokenizer - for actual model training (different from data formatting tokenizer above)
# 'AutoTokenizer.from_pretrained' is a method that loads a tokenizer from the Hugging Face Model Hub.
# 'model_id' is passed as an argument to specify which tokenizer to load.
# 'trust_remote_code' is set to True to trust the remote code in the tokenizer files.
# 'add_eos_token' is set to True to add an end-of-sentence token to the tokenizer.
# 'use_fast' is set to True to use the fast version of the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, add_eos_token=True, use_fast=True
)

# The padding token is set to the unknown token.
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
# The padding side is set to 'left', meaning that padding tokens will be added to the left (start) of the sequence.
# Left padding is preferred for causal LM training (different from 'right' padding used in data formatting)
tokenizer.padding_side = "left"

# 'AutoModelForCausalLM.from_pretrained' is a method that loads a pre-trained model for causal language modeling from the Hugging Face Model Hub.
# 'model_id' is passed as an argument to specify which model to load.
# 'torch_dtype' is set to the compute data type determined earlier.
# 'trust_remote_code' is set to True to trust the remote code in the model files.
# 'device_map' is passed as an argument to specify the device mapping for distributed training.
# 'attn_implementation' is set to the attention implementation determined earlier.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    device_map=DEVICE_MAP,
    attn_implementation=attn_implementation,
)


# This code block is used to define the training arguments for the model.

# 'TrainingArguments' is a class that holds the arguments for training a model.
# 'output_dir' is the directory where the model and its checkpoints will be saved.
# 'evaluation_strategy' is set to "steps", meaning that evaluation will be performed after a certain number of training steps.
# 'do_eval' is set to True, meaning that evaluation will be performed.
# 'optim' is set to "adamw_torch", meaning that the AdamW optimizer from PyTorch will be used.
# 'per_device_train_batch_size' and 'per_device_eval_batch_size' are set to 8, meaning that the batch size for training and evaluation will be 8 per device.
# 'gradient_accumulation_steps' is set to 4, meaning that gradients will be accumulated over 4 steps before performing a backward/update pass.
# 'log_level' is set to "debug", meaning that all log messages will be printed.
# 'save_strategy' is set to "epoch", meaning that the model will be saved after each epoch.
# 'logging_steps' is set to 100, meaning that log messages will be printed every 100 steps.
# 'learning_rate' is set to 1e-4, which is the learning rate for the optimizer.
# 'fp16' is set to the opposite of whether bfloat16 is supported on the current CUDA device.
# 'bf16' is set to whether bfloat16 is supported on the current CUDA device.
# 'eval_steps' is set to 100, meaning that evaluation will be performed every 100 steps.
# 'num_train_epochs' is set to 3, meaning that the model will be trained for 3 epochs.
# 'warmup_ratio' is set to 0.1, meaning that 10% of the total training steps will be used for the warmup phase.
# 'lr_scheduler_type' is set to "linear", meaning that a linear learning rate scheduler will be used.
# 'report_to' is set to "wandb", meaning that training and evaluation metrics will be reported to Weights & Biases.
# 'seed' is set to 42, which is the seed for the random number generator.

# LoraConfig object is created with the following parameters:
# 'r' (rank of the low-rank approximation) is set to 16,
# 'lora_alpha' (scaling factor) is set to 16,
# 'lora_dropout' dropout probability for Lora layers is set to 0.05,
# 'task_type' (set to TaskType.CAUSAL_LM indicating the task type),
# 'target_modules' (the modules to which LoRA is applied) choosing linear layers except the output layer..


args = TrainingArguments(
    output_dir=MODEL_NAME + "-LoRA",
    evaluation_strategy="steps",
    do_eval=True,
    optim="adamw_torch",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    log_level="debug",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=1e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    eval_steps=100,
    num_train_epochs=3,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    report_to="wandb",
    seed=42,
)

peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    task_type=TaskType.CAUSAL_LM,
    target_modules=TARGET_MODULES,
)

# 'model' is the model that will be trained.
# 'train_dataset' and 'eval_dataset' are the datasets that will be used for training and evaluation, respectively.
# 'peft_config' is the configuration for peft, which is used for instruction tuning.
# 'dataset_text_field' is set to "text", meaning that the 'text' field of the dataset will be used as the input for the model.
# 'max_seq_length' is set to 512, meaning that the maximum length of the sequences that will be fed to the model is 512 tokens.
# 'tokenizer' is the tokenizer that will be used to tokenize the input text.
# This uses the second tokenizer (training tokenizer) to convert text strings to token IDs
# 'args' are the training arguments that were defined earlier.

trainer = SFTTrainer(
    model=model,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=args,
)

# 'trainer.train()' is a method that starts the training of the model.
# It uses the training dataset, evaluation dataset, and training arguments that were provided when the trainer was initialized.
trainer.train()

# 'trainer.save_model()' is a method that saves the trained model locally.
# The model will be saved in the directory specified by 'output_dir' in the training arguments.
trainer.save_model()

trainer.push_to_hub(HF_MODEL_REPO)
