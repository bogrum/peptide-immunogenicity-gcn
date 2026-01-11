#!/bin/bash
# Process MD trajectory for a single peptide following Weber et al. 2024 methodology
# Usage: bash process_trajectory_single.sh PEPTIDE_NAME

set -e

PEPTIDE=$1
if [ -z "$PEPTIDE" ]; then
    echo "Usage: bash process_trajectory_single.sh PEPTIDE_NAME"
    exit 1
fi

# Get absolute paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYSTEM_DIR="$PROJECT_ROOT/md_data/systems/$PEPTIDE"
ANALYSIS_DIR="$PROJECT_ROOT/md_data/analysis/$PEPTIDE"

mkdir -p "$ANALYSIS_DIR"

echo "=========================================="
echo "Processing trajectory for: $PEPTIDE"
echo "=========================================="

cd "$SYSTEM_DIR"

# Check if trajectory exists
if [ ! -f "md.xtc" ]; then
    echo "ERROR: md.xtc not found for $PEPTIDE"
    exit 1
fi

echo "[1/8] Removing PBC artifacts (making molecules whole)..."
# Make molecules whole (remove PBC jumps)
echo "0" | gmx trjconv -f md.xtc -s md.tpr -o md_whole.xtc -pbc mol -ur compact 2>/dev/null

echo "[2/8] Extracting frames 30-100 ns (excluding equilibration)..."
# Extract 30-100 ns timeframe (exclude first 30 ns equilibration)
echo "0" | gmx trjconv -f md_whole.xtc -s md.tpr -o md_30-100ns.xtc -b 30000 -e 100000 2>/dev/null

# Use the processed trajectory for all analyses
TRAJ="md_30-100ns.xtc"

echo "[3/8] Calculating RMSD (no fitting)..."
# RMSD without fitting (just raw displacement from starting structure)
# Group 1 = Protein (MHC + peptide)
echo -e "1\n1" | gmx rms -f $TRAJ -s md.tpr -o "$ANALYSIS_DIR/rmsd.xvg" -fit none 2>/dev/null || echo "  Warning: RMSD calculation failed"

echo "[4/8] Calculating per-residue SASA..."
# SASA per residue
echo "1" | gmx sasa -f $TRAJ -s md.tpr -o "$ANALYSIS_DIR/sasa_total.xvg" -or "$ANALYSIS_DIR/sasa_per_residue.xvg" -surface "Protein" -output "Protein" 2>/dev/null || echo "  Warning: SASA calculation failed"

echo "[5/8] Calculating RMSF (per-residue fluctuations)..."
# RMSF - need to fit first for fluctuations
echo -e "4\n1" | gmx rmsf -f $TRAJ -s md.tpr -o "$ANALYSIS_DIR/rmsf.xvg" -res 2>/dev/null || echo "  Warning: RMSF calculation failed"

echo "[6/8] Extracting backbone dihedral angles..."
# Dihedral angles (phi, psi) for all residues
echo "1" | gmx rama -f $TRAJ -s md.tpr -o "$ANALYSIS_DIR/ramachandran.xvg" 2>/dev/null || echo "  Warning: Ramachandran calculation failed"

echo "[7/8] Calculating anchor distances (P2 and P9 to MHC)..."
# Create index groups for anchor positions
# First, get peptide residue numbers
FIRST_PEPTIDE_RES=$(grep -A 1 "peptide" index.ndx 2>/dev/null | tail -1 | awk '{print $1}' || echo "")

if [ -n "$FIRST_PEPTIDE_RES" ]; then
    # P2 is second residue of peptide, P9 is ninth
    P2_RES=$((FIRST_PEPTIDE_RES + 1))
    P9_RES=$((FIRST_PEPTIDE_RES + 8))

    # Calculate distances using gmx distance
    # Note: This is simplified - you may need to adjust based on actual MHC pocket residues
    echo "Creating anchor distance calculations..."
    # For now, calculate distance from peptide center to MHC alpha chain
    echo -e "1\n2" | gmx distance -f $TRAJ -s md.tpr -n index.ndx -oall "$ANALYSIS_DIR/anchor_distances.xvg" -select 'res $P2_RES; res $P9_RES' 2>/dev/null || echo "  Warning: Could not calculate anchor distances"
fi

echo "[8/8] Generating summary statistics..."
# Extract key statistics
cd "$ANALYSIS_DIR"

# Average SASA
if [ -f "sasa_total.xvg" ]; then
    AVG_SASA=$(grep -v "^[@#]" sasa_total.xvg | awk '{sum+=$2; n++} END {if(n>0) print sum/n}')
    echo "Average SASA: $AVG_SASA nm²" > summary.txt
fi

# Average RMSD
if [ -f "rmsd.xvg" ]; then
    AVG_RMSD=$(grep -v "^[@#]" rmsd.xvg | awk '{sum+=$2; n++} END {if(n>0) print sum/n}')
    echo "Average RMSD: $AVG_RMSD nm" >> summary.txt
fi

# Max RMSF
if [ -f "rmsf.xvg" ]; then
    MAX_RMSF=$(grep -v "^[@#]" rmsf.xvg | awk '{if($2>max) max=$2} END {print max}')
    echo "Max RMSF: $MAX_RMSF nm" >> summary.txt
fi

cd "$PROJECT_ROOT"

echo ""
echo "✓ Trajectory processing complete for $PEPTIDE"
echo "  Output directory: $ANALYSIS_DIR/"
echo "  - rmsd.xvg: RMSD over time (no fitting)"
echo "  - sasa_total.xvg: Total SASA over time"
echo "  - sasa_per_residue.xvg: Per-residue SASA"
echo "  - rmsf.xvg: Per-residue fluctuations"
echo "  - ramachandran.xvg: Backbone dihedral angles"
echo "  - summary.txt: Key statistics"
echo ""
