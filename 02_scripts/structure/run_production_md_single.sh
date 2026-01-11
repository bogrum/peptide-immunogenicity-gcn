#!/bin/bash
# Run production MD for a single peptide complex
# Usage: bash run_production_md_single.sh <peptide_name>

set -e

PEPTIDE=$1

if [ -z "$PEPTIDE" ]; then
    echo "Usage: $0 <peptide_name>"
    echo "Example: $0 FTSAVLLLL"
    exit 1
fi

WORKDIR="md_data/systems/${PEPTIDE}"

if [ ! -d "$WORKDIR" ]; then
    echo "Error: System not found: $WORKDIR"
    echo "Run preparation first!"
    exit 1
fi

cd "$WORKDIR"

echo "==========================================="
echo "Starting Production MD: $PEPTIDE"
echo "Duration: 100 ns"
echo "Workdir: $WORKDIR"
echo "==========================================="

# Prepare production MD
echo "Preparing production run..."
gmx grompp -f ../../mdp_files/md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 2

# Run production MD with optimized GPU configuration
echo "Starting MD simulation with GPU optimization..."
echo "Configuration: 4 MPI threads, 8 OpenMP threads, full GPU offload"
echo "This will take ~5 hours on RTX 5090 (~500 ns/day)"
echo ""

# GPU-optimized mdrun for RTX 5090
gmx mdrun -v -deffnm md \
    -ntmpi 4 -ntomp 8 \
    -nb gpu -pme gpu -bonded gpu -npme 1 \
    -pin on -noconfout \
    -dlb yes -notunepme

echo ""
echo "==========================================="
echo "✓ Production MD complete: $PEPTIDE"
echo "==========================================="
echo ""
echo "Output files:"
echo "  md.gro    - Final structure"
echo "  md.xtc    - Trajectory (compressed)"
echo "  md.edr    - Energy file"
echo "  md.log    - MD log file"
echo ""
