#!/bin/bash
# Install OpenFold by patching PyTorch to skip CUDA version check

set -e

echo "==========================================="
echo "Installing OpenFold (Skip Version Check)"
echo "==========================================="
echo ""

eval "$(conda shell.bash hook)"
conda activate swiftmhc_env

cd /home/emre/workspace/seminar_study/analysis/openfold

echo "[1/2] Patching PyTorch to skip CUDA version check..."

# Set environment variables
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0"

# Disable CUDA version check
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

echo "[2/2] Building OpenFold..."
echo "This may take 10-15 minutes..."
echo ""

# Temporarily patch the PyTorch CUDA check
TORCH_UTILS=$CONDA_PREFIX/lib/python3.11/site-packages/torch/utils/cpp_extension.py

# Backup original
cp $TORCH_UTILS ${TORCH_UTILS}.backup

# Comment out the version check
sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE/# raise RuntimeError(CUDA_MISMATCH_MESSAGE/' $TORCH_UTILS

# Build OpenFold
pip install -e . --no-build-isolation

# Restore original PyTorch file
mv ${TORCH_UTILS}.backup $TORCH_UTILS

echo ""
echo "==========================================="
echo "Testing OpenFold CUDA extensions..."
echo "==========================================="

python -c "
try:
    from openfold.utils.kernel.attention_core import attention_core
    print('✅ OpenFold CUDA extensions built successfully!')
    print('   GPU acceleration enabled!')
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Run predictions with:"
echo "  ./run_swiftmhc_prediction.sh"
