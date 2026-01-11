#!/bin/bash
# Test SwiftMHC Setup

eval "$(conda shell.bash hook)"
conda activate swiftmhc_env

# Add OpenFold and stub to PYTHONPATH
export PYTHONPATH="/home/emre/workspace/seminar_study/analysis:/home/emre/workspace/seminar_study/analysis/openfold:${PYTHONPATH}"
# Load CUDA stub before any OpenFold imports
export PYTHONSTARTUP="/home/emre/workspace/seminar_study/analysis/openfold_cuda_stub.py"

echo "=========================================="
echo "Testing SwiftMHC Installation"
echo "=========================================="
echo ""

echo "[1/4] Testing PyTorch + CUDA..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}'); print(f'  ✓ CUDA available: {torch.cuda.is_available()}')" || exit 1

echo ""
echo "[2/4] Testing OpenFold import..."
python -c "from openfold.np.residue_constants import restype_atom14_mask; print('  ✓ OpenFold imports successfully')" || exit 1

echo ""
echo "[3/4] Testing SwiftMHC import..."
python -c "import swiftmhc; print('  ✓ SwiftMHC imports successfully')" || exit 1

echo ""
echo "[4/4] Testing swiftmhc_predict command..."
cd /home/emre/workspace/seminar_study/analysis/swiftmhc-inference-main
swiftmhc_predict --help | head -10 || exit 1

echo ""
echo "=========================================="
echo "✅ All tests passed!"
echo "=========================================="
echo ""
echo "Ready to run predictions with:"
echo "  ./run_swiftmhc_prediction.sh"
