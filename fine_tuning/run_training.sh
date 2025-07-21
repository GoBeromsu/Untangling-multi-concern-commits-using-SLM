#!/bin/bash
#SBATCH --job-name=phi4_commit_sft_accelerate
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/phi4_training_%j.out
#SBATCH --error=logs/phi4_training_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

# Sheffield HPC Stanage - Multi-GPU A100 Training (2x80GB)
# Multi-Concern Commit Classification with Phi-4
# Setup: SFTTrainer + Accelerate YAML + Multi-GPU DDP

echo "Starting Phi-4 LoRA fine-tuning job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE MB"

module purge
module load GCCcore/12.3.0
module load CUDA/12.1.1
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate environment using 'source activate' instead of 'conda activate'
echo "🔧 Activating phi4_env..."
source activate phi4_env

# Set environment variables for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO  # Multi-GPU communication debugging

# Run distributed training with accelerate
echo "🔥 Starting multi-GPU training at $(date)"
accelerate launch --config_file accelerate_config.yaml train.py

echo "✅ Training completed at $(date)"

# Display basic job info
echo "📊 Job Summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,State,ExitCode

source deactivate 