#!/bin/bash

# Setup script for SLURM cluster
# Checks if conda environment exists, creates it if not, and activates it

set -e  # Exit on error

echo "=========================================="
echo "Embedding Service PoC - Cluster Setup"
echo "=========================================="

# Load required modules
echo "Loading SLURM modules..."
module load anaconda3/2024.06
module load cuda/12.1

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Environment name
ENV_NAME="embedding_poc"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Environment '${ENV_NAME}' already exists. Activating..."
    conda activate ${ENV_NAME}

    # Check if key packages are installed, install if missing
    echo "Verifying required packages..."
    python -c "import structlog" 2>/dev/null || {
        echo "Missing packages detected. Installing..."
        pip install --upgrade pip
        pip install vllm==0.11.0
        pip install fastapi uvicorn[standard] pydantic tyro
        pip install langchain langchain-community sentence-transformers
        pip install httpx numpy datasets
        pip install psutil pynvml
        pip install structlog colorama
        pip install matplotlib seaborn
    }
else
    echo ""
    echo "Environment '${ENV_NAME}' not found. Creating from environment.yml..."
    conda env create -f environment.yml
    echo ""
    echo "Activating newly created environment..."
    conda activate ${ENV_NAME}

    # Sometimes pip packages from yaml don't install properly, install them explicitly
    echo ""
    echo "Ensuring pip packages are installed..."
    pip install --upgrade pip
    pip install vllm==0.11.0
    pip install fastapi uvicorn[standard] pydantic tyro
    pip install langchain langchain-community sentence-transformers
    pip install httpx numpy datasets
    pip install psutil pynvml
    pip install structlog colorama
    pip install matplotlib seaborn
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Current environment: ${CONDA_DEFAULT_ENV}"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Checking CUDA/GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
echo ""
