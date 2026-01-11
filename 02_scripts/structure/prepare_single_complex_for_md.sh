#!/bin/bash
# Prepare a single peptide-MHC complex for GROMACS MD simulation
# Usage: bash prepare_single_complex_for_md.sh <input_pdb> <output_name>

set -e  # Exit on error

INPUT_PDB=$1
OUTPUT_NAME=$2

if [ -z "$INPUT_PDB" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Usage: $0 <input_pdb> <output_name>"
    exit 1
fi

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Convert to absolute path if relative
if [[ "$INPUT_PDB" != /* ]]; then
    INPUT_PDB="$PROJECT_ROOT/$INPUT_PDB"
fi

if [ ! -f "$INPUT_PDB" ]; then
    echo "Error: Input PDB file not found: $INPUT_PDB"
    exit 1
fi

# Create working directory
WORKDIR="$PROJECT_ROOT/md_data/systems/${OUTPUT_NAME}"
MDP_DIR="$PROJECT_ROOT/md_data/mdp_files"
mkdir -p "$WORKDIR"

echo "==========================================="
echo "Preparing: $OUTPUT_NAME"
echo "Input: $INPUT_PDB"
echo "Output: $WORKDIR"
echo "==========================================="

cd "$WORKDIR"

# Copy input PDB
cp "$INPUT_PDB" complex.pdb

echo "[1/7] Processing PDB structure..."
# Generate topology using AMBER99SB-ILDN force field
echo "1" | gmx pdb2gmx -f complex.pdb -o processed.gro -water tip3p -ignh -ff amber99sb-ildn

echo "[2/7] Defining simulation box..."
# Create a cubic box with 1.0 nm distance from protein to box edge
gmx editconf -f processed.gro -o boxed.gro -c -d 1.0 -bt cubic

echo "[3/7] Solvating the system..."
# Fill box with water
gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top

echo "[4/7] Adding ions..."
# Create TPR file for adding ions
gmx grompp -f "$MDP_DIR/ions.mdp" -c solvated.gro -p topol.top -o ions.tpr -maxwarn 2

# Add ions to neutralize the system (automatically selects SOL group)
echo "SOL" | gmx genion -s ions.tpr -o ionized.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15

echo "[5/7] Energy minimization..."
gmx grompp -f "$MDP_DIR/minim.mdp" -c ionized.gro -p topol.top -o em.tpr -maxwarn 2
# Minimization: CPU only (steepest descent doesn't support GPU PME)
gmx mdrun -v -deffnm em -ntmpi 4 -ntomp 8

echo "[6/7] NVT equilibration (GPU accelerated)..."
gmx grompp -f "$MDP_DIR/nvt.mdp" -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 2
gmx mdrun -v -deffnm nvt -ntmpi 4 -ntomp 8 -nb gpu -pme gpu -bonded gpu -npme 1 -pin on

echo "[7/7] NPT equilibration (GPU accelerated)..."
gmx grompp -f "$MDP_DIR/npt.mdp" -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 2
gmx mdrun -v -deffnm npt -ntmpi 4 -ntomp 8 -nb gpu -pme gpu -bonded gpu -npme 1 -pin on

echo "==========================================="
echo "✓ Preparation complete: $OUTPUT_NAME"
echo "==========================================="
echo ""
echo "System ready for production MD!"
echo "Next: Run production simulation with md.mdp"
echo ""
