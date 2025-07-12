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

# Load required modules (HPC recommended versions)
module load GCC/12.3.0
module load CUDA/12.4.0
module load cuDNN/8.9.2.26-CUDA-12.1.1  # HPC requirement for GPU PyTorch
module load Anaconda3/2022.05  # HPC recommended stable version  

# Setup conda environment (HPC best practice for PyTorch)
if conda env list | grep -q "phi4_env"; then
    echo "🗑️ Removing existing phi4_env..."
    conda remove -n phi4_env --all -y
fi

echo "🏗️ Creating conda environment..."
conda create -n phi4_env python=3.11 -y
source activate phi4_env  # HPC requires 'source activate' instead of 'conda activate'

# Upgrade pip
pip install --upgrade pip

# Install dependencies (HPC Stanage conda-first approach)
echo "Installing dependencies..."

# Install core ML packages via conda (HPC strongly recommended)
echo "📦 Installing conda packages..."
conda install -c conda-forge numpy=1.24.4 -y  # NumPy 1.x compatibility
conda install -c pytorch -c nvidia pytorch=2.1.0 torchvision torchaudio pytorch-cuda=12.1 -y  # Official PyTorch
conda install -c conda-forge scipy pandas -y  # Scientific computing
conda install -c conda-forge accelerate -y  # Available in conda-forge

# Install specialized ML packages via pip (not in conda)
echo "📦 Installing pip packages..."
pip install wheel setuptools  # Essential build tools
pip install transformers==4.48.1
pip install datasets  
pip install peft==0.14.0
pip install trl  # Not available in conda

# Note: flash-attn removed due to glibc compatibility - using PyTorch SDPA fallback

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

# Monitor GPU usage in background (HPC recommended)
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.total,memory.used,memory.free --format=csv -l 2 > gpu_stats_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Run training
echo "🔥 Starting LoRA fine-tuning at $(date)"
python models/train.py

# Stop monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

echo "Training completed at $(date)"
echo "Check gpu_stats_${SLURM_JOB_ID}.log for GPU utilization"

# Resource usage summary (HPC recommended)
echo "📊 Resource Usage Summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,Elapsed,State
echo "📈 GPU Utilization Summary:"
echo "To view GPU stats: head -5 gpu_stats_${SLURM_JOB_ID}.log"
echo "For efficiency report: seff ${SLURM_JOB_ID}"

# Clean up  
source deactivate  # HPC conda deactivation