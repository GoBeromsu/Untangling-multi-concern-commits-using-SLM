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

# Step 1: Install PyTorch 2.2+ (required for flash-attn)
echo "📦 Installing PyTorch 2.2+ with CUDA 12.1 support..."
pip install "torch>=2.2.0"

# Step 2: Install flash-attn prerequisites
echo "📦 Installing flash-attn prerequisites..."
pip install packaging ninja

# Step 3: Install prebuilt flash-attn wheel (manylinux2014, GLIBC 2.17)
# Wheel repo: https://github.com/mjun0812/flash-attention-prebuild-wheels
# Choose build matching: flash_attn-2.7.3+pt222cu121cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
FLASH_ATTN_WHEEL_URL = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.4/flash_attn-2.7.3%2Bpt222cu121cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
echo "📦 Downloading & installing flash-attn wheel..."
if ! pip install --no-cache-dir "$FLASH_ATTN_WHEEL_URL"; then
    echo "❌ Failed to install flash-attn wheel. Falling back to source build."
    export FLASH_ATTENTION_FORCE_CUDA=1
    export TORCH_CUDA_ARCH_LIST="80" # A100 architecture
    echo "📦 Building flash-attn from source as fallback... (this may take a while)"
    pip install flash-attn --no-build-isolation  --use-pep517
fi

# Step 4: Install remaining ML dependencies
 echo "📦 Installing ML dependencies..."
pip install -r requirements.txt

# Verify flash-attn installation
python - <<'PY' > logs/flash_attn_verify_${SLURM_JOB_ID}.log
import flash_attn, torch, os
print(
    f"flash-attn {flash_attn.__version__} ✓  "
    f"PyTorch {torch.__version__}  "
    f"CUDA {torch.version.cuda}  "
    f"CC {os.getenv('TORCH_CUDA_ARCH_LIST', 'default')}"
)
PY

echo "✅ Environment setup completed successfully!"
echo "To activate the environment manually: source activate phi4_env"

source deactivate 