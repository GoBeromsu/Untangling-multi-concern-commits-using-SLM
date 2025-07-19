#!/bin/bash
#SBATCH --job-name=phi4-env-setup
#SBATCH --time=0:60:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/phi4_env_setup_%j.out
#SBATCH --error=logs/phi4_env_setup_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

# Sheffield HPC Stanage - Environment Setup for Phi-4 Fine-tuning
# Multi-Concern Commit Classification with Phi-4

echo "Setting up Phi-4 LoRA fine-tuning environment..."

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
if ! conda env create -f environment.yml; then
    echo "❌ Failed to create conda environment. Exiting..."
    exit 1
fi

# Activate environment using 'source activate' instead of 'conda activate'
# Sheffield HPC requirement: Due to Anaconda being installed as a module,
# must use 'source' command instead of 'conda' when activating environments
# Reference: https://docs.hpc.shef.ac.uk/en/latest/stanage/software/apps/python.html
echo "🔧 Activating phi4_env..."
source activate phi4_env

# Step 1: Install PyTorch first (required for flash-attn compilation)
echo "📦 Installing PyTorch with CUDA 12.1 support..."
pip install torch>=2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install remaining ML dependencies that depend on torch
echo "📦 Installing ML dependencies..."
pip install -r requirements.txt

echo "✅ Environment setup completed successfully!"
echo "To activate the environment manually: source activate phi4_env"

source deactivate 