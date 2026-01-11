#!/bin/bash
# Fix CUDA version mismatch and install OpenFold properly

set -e

echo "==========================================="
echo "Installing CUDA 12.4 Toolkit & OpenFold"
echo "==========================================="
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate swiftmhc_env

echo "[1/4] Installing CUDA 12.4 toolkit from conda..."
echo "This provides a local CUDA that matches PyTorch"
echo ""

# Install CUDA 12.4 toolkit in the conda environment
conda install -c nvidia cuda-toolkit=12.4 -y

echo ""
echo "[2/4] Installing ninja build system..."
# Ninja is already installed, but let's make sure
conda install ninja -y

echo ""
echo "[3/4] Building OpenFold with CUDA extensions..."
echo "This may take 10-15 minutes..."
echo ""

# Navigate to OpenFold directory
cd /home/emre/workspace/seminar_study/analysis/openfold

# Set CUDA environment variables to use conda's CUDA
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set architecture for RTX 5090 (Blackwell - compute capability 12.0)
# Use 9.0 as fallback since 12.0 is very new and may not be fully supported yet
export TORCH_CUDA_ARCH_LIST="9.0"

# Build and install OpenFold
pip install -e . --no-build-isolation

echo ""
echo "[4/4] Testing OpenFold CUDA extensions..."
python -c "
try:
    from openfold.utils.kernel.attention_core import attention_core
    print('✅ OpenFold CUDA extensions built successfully!')
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

echo ""
echo "==========================================="
echo "✅ OpenFold installed with CUDA support!"
echo "==========================================="
echo ""
echo "Now SwiftMHC will use optimized CUDA kernels!"
echo ""
echo "Run predictions with:"
echo "  ./run_swiftmhc_prediction.sh"
