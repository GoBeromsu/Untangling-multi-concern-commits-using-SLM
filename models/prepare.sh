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

# Load required HPC modules (latest versions for optimal bf16 support)
module load GCC/13.2.0
module load CUDA/12.4.0  
module load cuDNN/9.0.0.312-CUDA-12.4.0
module load Anaconda3/2023.09  

# Setup conda environment using environment.yml (pure conda approach)
if conda env list | grep -q "phi4_env"; then
    echo "🗑️ Removing existing phi4_env..."
    conda remove -n phi4_env --all -y
fi

echo "🏗️ Creating conda environment from environment.yml..."
conda env create -f models/environment.yml
source activate phi4_env  # HPC requires 'source activate' instead of 'conda activate'

echo "📦 All dependencies installed via conda environment.yml"

# Note: flash-attn removed due to glibc compatibility - using PyTorch SDPA fallback
# Set environment variables for optimal GPU performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Start GPU monitoring in background
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.total,memory.used,memory.free --format=csv -l 2 > gpu_stats_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Run training script
echo "🔥 Starting LoRA fine-tuning at $(date)"
python models/train.py

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

echo "Training completed at $(date)"
echo "Check gpu_stats_${SLURM_JOB_ID}.log for GPU utilization"

# Display resource usage summary
echo "📊 Resource Usage Summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,Elapsed,State
echo "📈 GPU Utilization Summary:"
echo "To view GPU stats: head -5 gpu_stats_${SLURM_JOB_ID}.log"
echo "For efficiency report: seff ${SLURM_JOB_ID}"

# Clean up conda environment
source deactivate