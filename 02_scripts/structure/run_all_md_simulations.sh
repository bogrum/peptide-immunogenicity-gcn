#!/bin/bash
# Master script to prepare and run MD simulations for all 30 peptide-MHC complexes
# Can be run in tmux for background execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=============================================="
echo "  MD Simulation Pipeline for 30 Peptides"
echo "=============================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Results: 03_results/swiftmhc_output/"
echo "MD workspace: md_data/"
echo ""

# Create necessary directories
mkdir -p md_data/systems
mkdir -p md_data/logs

# Get list of all PDB files
PDB_FILES=(03_results/swiftmhc_output/HLA-Ax02_01-*.pdb)
TOTAL=${#PDB_FILES[@]}

echo "Found $TOTAL peptide-MHC complexes"
echo ""
echo "Simulation stages per complex:"
echo "  1. Structure preparation (pdb2gmx)"
echo "  2. Solvation and ionization"
echo "  3. Energy minimization (~5 min)"
echo "  4. NVT equilibration (~5 min)"
echo "  5. NPT equilibration (~5 min)"
echo "  6. Production MD 10ns (~2-3 hours on GPU)"
echo ""
echo "Estimated total time: ~3 hours per complex"
echo "Total for all 30: ~90 hours (will run in parallel)"
echo ""

read -p "Continue with preparation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Process each complex
COUNTER=0
for PDB in "${PDB_FILES[@]}"; do
    COUNTER=$((COUNTER + 1))

    # Extract peptide sequence from filename
    BASENAME=$(basename "$PDB" .pdb)
    PEPTIDE=$(echo "$BASENAME" | sed 's/HLA-Ax02_01-//')

    echo "=============================================="
    echo "[$COUNTER/$TOTAL] Processing: $PEPTIDE"
    echo "=============================================="

    # Prepare the system
    LOG_FILE="md_data/logs/${PEPTIDE}_preparation.log"

    bash 02_scripts/structure/prepare_single_complex_for_md.sh \
        "$PDB" "$PEPTIDE" 2>&1 | tee "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ $PEPTIDE preparation complete"
    else
        echo "✗ $PEPTIDE preparation failed - check $LOG_FILE"
    fi
    echo ""
done

echo "=============================================="
echo "  Preparation Complete!"
echo "=============================================="
echo ""
echo "All systems prepared in: md_data/systems/"
echo ""
echo "Next steps:"
echo "1. Review preparation logs: md_data/logs/"
echo "2. Run production MD simulations"
echo ""
echo "To run production MD for all complexes:"
echo "  bash 02_scripts/structure/run_production_md_all.sh"
echo ""
