#!/bin/bash
#SBATCH --job-name=phi4_commit_sft
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32 
#SBATCH --mem=256G
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
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE MB"

# Create logs directory
mkdir -p logs

module purge
module load GCCcore/12.3.0
module load CUDA/12.1.1
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Remove existing environment if exists
if conda env list | grep -q "phi4_env"; then
    echo "🗑️ Removing existing phi4_env..."
    conda remove -n phi4_env --all -y
fi

# Create conda environment
echo "🏗️ Creating conda environment..."
conda env create -f environment.yml

# Activate environment
echo "🔧 Activating phi4_env..."
conda activate phi4_env

# Install pip dependencies
echo "📦 Installing ML dependencies..."
pip install -r requirements.txt

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Quick verification
echo "🔍 Verifying setup..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run training
echo "🔥 Starting training at $(date)"
python train.py

echo "✅ Training completed at $(date)"

# Display basic job info
echo "📊 Job Summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,State,ExitCode

# Cleanup
conda deactivate