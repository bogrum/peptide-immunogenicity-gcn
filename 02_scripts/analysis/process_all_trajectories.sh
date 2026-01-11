#!/bin/bash
# Process all 30 MD trajectories following Weber et al. 2024 methodology
# This extracts features needed for unsupervised learning (Markov models)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  Trajectory Processing for All Peptides"
echo "  Following Weber et al. 2024 Methodology"
echo "=============================================="
echo ""
echo "Processing details:"
echo "  - Time window: 30-100 ns (exclude 30 ns equilibration)"
echo "  - PBC correction: molecules made whole"
echo "  - Keep explicit solvent for SASA calculations"
echo ""

# Create analysis directory
mkdir -p md_data/analysis

# Get list of all peptides with completed MD
PEPTIDES=()
for dir in md_data/systems/*/; do
    peptide=$(basename "$dir")
    if [ -f "$dir/md.xtc" ] && [ -f "$dir/md.tpr" ]; then
        # Check if completed (has Performance marker)
        if tail -100 "$dir/md.log" 2>/dev/null | grep -q "Performance:"; then
            PEPTIDES+=("$peptide")
        fi
    fi
done

TOTAL=${#PEPTIDES[@]}
echo "Found $TOTAL completed simulations to process"
echo ""

# Process each peptide
SUCCESS=0
FAILED=0

for i in "${!PEPTIDES[@]}"; do
    peptide="${PEPTIDES[$i]}"
    n=$((i + 1))

    echo "[$n/$TOTAL] Processing $peptide..."

    if bash 02_scripts/analysis/process_trajectory_single.sh "$peptide"; then
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ Failed to process $peptide"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================="
echo "  Processing Complete"
echo "=============================================="
echo "  Success: $SUCCESS/$TOTAL"
echo "  Failed: $FAILED/$TOTAL"
echo ""
echo "Output location: md_data/analysis/"
echo ""

# Create a summary file with all statistics
echo "Generating combined summary..."
SUMMARY_FILE="md_data/analysis/all_peptides_summary.csv"
echo "Peptide,Avg_SASA_nm2,Avg_RMSD_nm,Max_RMSF_nm" > "$SUMMARY_FILE"

for peptide in "${PEPTIDES[@]}"; do
    if [ -f "md_data/analysis/$peptide/summary.txt" ]; then
        SASA=$(grep "Average SASA" "md_data/analysis/$peptide/summary.txt" | awk '{print $3}')
        RMSD=$(grep "Average RMSD" "md_data/analysis/$peptide/summary.txt" | awk '{print $3}')
        RMSF=$(grep "Max RMSF" "md_data/analysis/$peptide/summary.txt" | awk '{print $3}')
        echo "$peptide,$SASA,$RMSD,$RMSF" >> "$SUMMARY_FILE"
    fi
done

echo "✓ Combined summary saved to: $SUMMARY_FILE"
echo ""
echo "Next steps:"
echo "  1. Extract dihedral angles for Markov model building"
echo "  2. Calculate hydrophobic vs hydrophilic SASA per residue"
echo "  3. Identify anchor residues and calculate P2/P9 dynamics"
echo "  4. Build molecular graphs (8 Å cutoff) for GCN training"
echo ""
