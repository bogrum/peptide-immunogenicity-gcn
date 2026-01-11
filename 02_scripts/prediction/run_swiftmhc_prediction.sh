#!/bin/bash
# SwiftMHC Prediction Runner Script
# This script sets up the environment and runs SwiftMHC predictions

set -e  # Exit on error

echo "==========================================="
echo "SwiftMHC Prediction Runner"
echo "==========================================="
echo ""

# Activate conda environment
echo "[1/4] Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate swiftmhc_env

# Add OpenFold to PYTHONPATH (patched with CPU fallback)
echo "[2/4] Setting up Python path..."
export PYTHONPATH="/home/emre/workspace/seminar_study/analysis/openfold:${PYTHONPATH}"

# Force CPU mode (RTX 5090 sm_120 not supported by current PyTorch)
export CUDA_VISIBLE_DEVICES=""

# Navigate to SwiftMHC directory
cd /home/emre/workspace/seminar_study/analysis/swiftmhc-inference-main

echo "[3/4] Running SwiftMHC prediction (CPU mode)..."
echo ""
echo "  Input: our_peptides.csv (30 peptides)"
echo "  Model: trained-models/8k-trained-model.pth"
echo "  HLA: data/HLA-A0201-from-3MRD.hdf5"
echo "  Output: results_our_peptides/"
echo ""
echo "NOTE: Running in CPU mode because RTX 5090 (sm_120) requires newer PyTorch"
echo "This may take 5-10 minutes..."
echo ""

# Run SwiftMHC with structure building
swiftmhc_predict \
    --num-builders 1 \
    --batch-size 8 \
    trained-models/8k-trained-model.pth \
    our_peptides.csv \
    data/HLA-A0201-from-3MRD.hdf5 \
    results_our_peptides

echo ""
echo "==========================================="
echo "✅ Prediction Complete!"
echo "==========================================="
echo ""
echo "Results saved in: results_our_peptides/"
echo "  - results.csv: Binding affinities"
echo "  - *.pdb: 3D structures (30 files)"
echo ""
echo "Next steps:"
echo "  1. View results: cat results_our_peptides/results.csv"
echo "  2. Visualize structures: pymol results_our_peptides/*.pdb"
echo "  3. Run MD simulations with GROMACS"
