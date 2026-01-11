#!/bin/bash
# Prepare all 30 systems in tmux (can run in background)

SESSION_NAME="md_preparation"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=============================================="
echo "  Preparing 30 Peptide-MHC Systems for MD"
echo "=============================================="
echo ""

# Check if tmux session exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists"
    read -p "Kill and restart? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
    else
        echo "Attach with: tmux attach -t $SESSION_NAME"
        exit 0
    fi
fi

# Create tmux session
echo "Creating tmux session..."
tmux new-session -d -s "$SESSION_NAME"

# Run preparation script in tmux
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_ROOT" C-m
tmux send-keys -t "$SESSION_NAME" "echo '============================================== '" C-m
tmux send-keys -t "$SESSION_NAME" "echo '  MD Preparation for 30 Peptides'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '============================================== '" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m

# Process each PDB file
for PDB in 03_results/swiftmhc_output/HLA-Ax02_01-*.pdb; do
    BASENAME=$(basename "$PDB" .pdb)
    PEPTIDE=$(echo "$BASENAME" | sed 's/HLA-Ax02_01-//')

    tmux send-keys -t "$SESSION_NAME" "echo '==========================================='" C-m
    tmux send-keys -t "$SESSION_NAME" "echo 'Preparing: $PEPTIDE'" C-m
    tmux send-keys -t "$SESSION_NAME" "echo '==========================================='" C-m
    tmux send-keys -t "$SESSION_NAME" "bash 02_scripts/structure/prepare_single_complex_for_md.sh $PDB $PEPTIDE 2>&1 | tee md_data/logs/${PEPTIDE}_prep.log" C-m
done

# Final message
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME" "echo '============================================== '" C-m
tmux send-keys -t "$SESSION_NAME" "echo '✓ All 30 systems prepared!'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '============================================== '" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Results in: md_data/systems/'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Logs in: md_data/logs/'" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Next step: Run production MD'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '  bash 02_scripts/structure/run_md_with_tmux.sh'" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Press Ctrl+D to close this window'" C-m

echo ""
echo "=============================================="
echo "Preparation started in tmux session!"
echo "=============================================="
echo ""
echo "Session: $SESSION_NAME"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach (let it run in background):"
echo "  Ctrl+b then d"
echo ""
echo "Estimated time: ~2.5 hours for all 30"
echo "(~5 min per peptide)"
echo ""
echo "Attaching to session in 3 seconds..."
sleep 3
tmux attach -t "$SESSION_NAME"
