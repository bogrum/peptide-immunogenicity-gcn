#!/bin/bash
# Master script to run entire analysis pipeline
# Run this AFTER you have completed your MD simulations

set -e  # Exit on error

echo "========================================"
echo "HLA-A2 Immunogenicity Analysis Pipeline"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required!"
    exit 1
fi

echo "Step 0: Checking dependencies..."
python3 -c "import pandas, numpy, matplotlib" 2>/dev/null || {
    echo "WARNING: Some dependencies missing. Install with:"
    echo "  pip install -r requirements.txt"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

echo ""
echo "========================================"
echo "Step 1: Structure Preparation"
echo "========================================"
python3 scripts/01_prepare_structures.py
echo "✓ Structure preparation complete"
echo ""
echo "Next: Model peptide structures following peptides/structures/INSTRUCTIONS.md"
echo "      Then run your GROMACS MD simulations (100 ns each)"
echo ""
read -p "Have you completed MD simulations? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Complete MD simulations first, then run this script again."
    exit 0
fi

echo ""
echo "========================================"
echo "Step 2: MD Analysis"
echo "========================================"
python3 scripts/02_analyze_md.py
if [ $? -eq 0 ]; then
    echo "✓ MD analysis complete"
else
    echo "✗ MD analysis failed - check logs"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 3: Visualization"
echo "========================================"
python3 scripts/03_visualize_results.py
if [ $? -eq 0 ]; then
    echo "✓ Visualization complete"
    echo "Check results/figures/ for plots"
else
    echo "✗ Visualization failed - check logs"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 4: Build Molecular Graphs (Optional)"
echo "========================================"
read -p "Build graphs for ML? This may take time. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 scripts/04_build_molecular_graphs.py
    if [ $? -eq 0 ]; then
        echo "✓ Graph building complete"
    else
        echo "✗ Graph building failed - check logs"
    fi
fi

echo ""
echo "========================================"
echo "Step 5: Train GCN Model (Optional)"
echo "========================================"
read -p "Train GCN model? Requires PyTorch. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 scripts/05_train_gcn_model.py
    if [ $? -eq 0 ]; then
        echo "✓ Model training complete"
    else
        echo "✗ Model training failed - check logs"
    fi
fi

echo ""
echo "========================================"
echo "PIPELINE COMPLETE!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Figures: results/figures/"
echo "  - Statistics: results/comparative_statistics.csv"
echo "  - Logs: results/logs/"
echo ""
echo "For your presentation, use:"
echo "  📊 results/figures/presentation_summary.png"
echo ""
echo "Good luck with your seminar! 🎓"
echo ""
