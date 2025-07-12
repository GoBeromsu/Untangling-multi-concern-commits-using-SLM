#!/bin/bash
#SBATCH --job-name=phi4_commit_sft
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16 
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/phi4_training_%j.out
#SBATCH --error=logs/phi4_training_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

# Sheffield HPC Stanage - A100 GPU Training
# Multi-Concern Commit Classification with Phi-4

echo "Starting Phi-4 LoRA fine-tuning job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Create logs directory
mkdir -p logs

# Load required modules
module load GCC/12.3.0
module load CUDA/12.4.0
module load Python/3.11.3-GCCcore-12.3.0

# Setup Python environment
python -m venv phi4_env
source phi4_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.48.1
pip install datasets
pip install peft==0.14.0
pip install trl
pip install accelerate==1.3.0
pip install bitsandbytes
pip install flash-attn --no-build-isolation

# Verify GPU setup
echo "🔍 Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Set environment variables for optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Monitor GPU usage in background
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu --format=csv -l 30 > logs/gpu_usage_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Run training
echo "🔥 Starting LoRA fine-tuning at $(date)"
python models/train.py

# Stop monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

echo "Training completed at $(date)"
echo "Check logs/gpu_usage_${SLURM_JOB_ID}.log for GPU utilization"

# Resource usage summary
echo "📊 Resource Usage Summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,Elapsed,State

# Clean up
deactivate