#!/bin/bash
# Run production MD for all prepared systems
# Runs sequentially (or can be modified for parallel execution)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  Production MD for All 30 Peptides"
echo "=============================================="
echo ""

# Find all prepared systems
SYSTEMS=(md_data/systems/*)
TOTAL=${#SYSTEMS[@]}

if [ $TOTAL -eq 0 ]; then
    echo "Error: No prepared systems found in md_data/systems/"
    echo "Run preparation first: bash 02_scripts/structure/run_all_md_simulations.sh"
    exit 1
fi

echo "Found $TOTAL prepared systems"
echo "Each simulation will take ~2-3 hours on GPU"
echo "Total estimated time: ~60-90 hours (running sequentially)"
echo ""
echo "TIP: To run faster, use the tmux parallel script!"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

COUNTER=0
for SYSTEM_DIR in "${SYSTEMS[@]}"; do
    COUNTER=$((COUNTER + 1))
    PEPTIDE=$(basename "$SYSTEM_DIR")

    echo ""
    echo "=============================================="
    echo "[$COUNTER/$TOTAL] Running MD: $PEPTIDE"
    echo "=============================================="

    bash 02_scripts/structure/run_production_md_single.sh "$PEPTIDE"

    echo "✓ Completed: $PEPTIDE ($COUNTER/$TOTAL)"
done

echo ""
echo "=============================================="
echo "  All MD Simulations Complete!"
echo "=============================================="
echo ""
echo "Results in: md_data/systems/*/md.xtc"
echo ""
