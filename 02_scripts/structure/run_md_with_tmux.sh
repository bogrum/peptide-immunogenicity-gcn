#!/bin/bash
# Run MD simulations in parallel using tmux
# Each peptide runs in its own tmux window
# You can detach and let them run in the background

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

SESSION_NAME="md_simulations"
PARALLEL_JOBS=1  # Run 1 simulation at a time (single GPU)

echo "=============================================="
echo "  TMUX-Based Parallel MD Simulations"
echo "=============================================="
echo ""
echo "Session name: $SESSION_NAME"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed"
    echo "Install with: sudo apt install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Warning: Session '$SESSION_NAME' already exists"
    read -p "Kill existing session and start fresh? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
    else
        echo "Attach to existing session with: tmux attach -t $SESSION_NAME"
        exit 0
    fi
fi

# Get list of prepared systems
SYSTEMS=(md_data/systems/*)
TOTAL=${#SYSTEMS[@]}

if [ $TOTAL -eq 0 ]; then
    echo "Error: No prepared systems found"
    echo "Run preparation first: bash 02_scripts/structure/run_all_md_simulations.sh"
    exit 1
fi

echo "Found $TOTAL prepared systems"
echo ""
echo "Strategy:"
echo "  - Create tmux session with multiple windows"
echo "  - Each window runs one MD simulation"
echo "  - Run $PARALLEL_JOBS jobs in parallel"
echo "  - When one finishes, start the next"
echo ""
echo "You can:"
echo "  - Detach: Ctrl+b then d"
echo "  - Reattach: tmux attach -t $SESSION_NAME"
echo "  - View windows: Ctrl+b then w"
echo ""

read -p "Start MD simulations? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create tmux session
echo "Creating tmux session..."
tmux new-session -d -s "$SESSION_NAME" -n "main"

# Send welcome message to main window
tmux send-keys -t "$SESSION_NAME:main" "echo '═══════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '  MD Simulation Controller'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '═══════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo 'Running $TOTAL MD simulations in parallel'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo 'Parallel jobs: $PARALLEL_JOBS'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo 'Monitor individual simulations:'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '  Ctrl+b then w  - List all windows'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '  Ctrl+b then n  - Next window'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '  Ctrl+b then p  - Previous window'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo 'Detach from session: Ctrl+b then d'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo 'Reattach: tmux attach -t $SESSION_NAME'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo ''" C-m

# Create windows and start simulations
COUNTER=0
ACTIVE_JOBS=0

for SYSTEM_DIR in "${SYSTEMS[@]}"; do
    PEPTIDE=$(basename "$SYSTEM_DIR")
    COUNTER=$((COUNTER + 1))

    # Wait if we have too many parallel jobs
    while [ $ACTIVE_JOBS -ge $PARALLEL_JOBS ]; do
        sleep 10
        # Check how many are still running
        ACTIVE_JOBS=$(tmux list-windows -t "$SESSION_NAME" -F "#{window_name}" | grep -v "main" | wc -l)
    done

    echo "Starting MD simulation for $PEPTIDE ($COUNTER/$TOTAL)..."

    # Create new window for this peptide
    tmux new-window -t "$SESSION_NAME" -n "$PEPTIDE"

    # Run the MD simulation in this window
    tmux send-keys -t "$SESSION_NAME:$PEPTIDE" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$SESSION_NAME:$PEPTIDE" "bash 02_scripts/structure/run_production_md_single.sh $PEPTIDE" C-m
    tmux send-keys -t "$SESSION_NAME:$PEPTIDE" "echo ''" C-m
    tmux send-keys -t "$SESSION_NAME:$PEPTIDE" "echo '✓ Simulation complete - this window will stay open'" C-m
    tmux send-keys -t "$SESSION_NAME:$PEPTIDE" "echo 'Press Ctrl+b then d to detach'" C-m

    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))

    # Small delay between starting jobs
    sleep 2
done

echo ""
echo "=============================================="
echo "  All simulations queued!"
echo "=============================================="
echo ""
echo "Tmux session '$SESSION_NAME' created with $TOTAL windows"
echo ""
echo "To monitor:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "Then use:"
echo "  Ctrl+b w  - List all windows"
echo "  Ctrl+b n  - Next window"
echo "  Ctrl+b p  - Previous window"
echo "  Ctrl+b d  - Detach (simulations keep running)"
echo ""
echo "To check status later:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To kill all simulations:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""

# Attach to the session
echo "Attaching to session in 3 seconds..."
sleep 3
tmux attach -t "$SESSION_NAME"
