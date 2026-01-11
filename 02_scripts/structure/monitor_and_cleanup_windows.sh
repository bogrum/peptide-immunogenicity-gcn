#!/bin/bash
# Monitor tmux windows and close them when MD simulations finish
# This allows the main script to continue launching new jobs

SESSION_NAME="md_simulations"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Starting window cleanup monitor for session: $SESSION_NAME"
echo "This will auto-close windows when simulations complete"
echo ""

while true; do
    # Get list of non-main windows
    WINDOWS=$(tmux list-windows -t "$SESSION_NAME" -F "#{window_name}" 2>/dev/null | grep -v "^main$" || true)

    if [ -z "$WINDOWS" ]; then
        sleep 10
        continue
    fi

    # Check each window
    for WINDOW in $WINDOWS; do
        # Check if there's a gmx process running in this window
        # Get the pane PID and check if any child process is gmx
        PANE_PID=$(tmux list-panes -t "$SESSION_NAME:$WINDOW" -F "#{pane_pid}" 2>/dev/null || true)

        if [ -n "$PANE_PID" ]; then
            # Check if gmx is running under this pane
            GMX_RUNNING=$(pgrep -P "$PANE_PID" gmx 2>/dev/null || true)

            if [ -z "$GMX_RUNNING" ]; then
                # Check if simulation actually completed (look for completion message in log)
                if [ -f "md_data/systems/$WINDOW/md.log" ]; then
                    # Check if log shows completion (Performance: line appears at the end)
                    if tail -100 "md_data/systems/$WINDOW/md.log" | grep -q "Performance:"; then
                        echo "[$(date +%H:%M:%S)] ✓ $WINDOW completed - closing window"
                        tmux kill-window -t "$SESSION_NAME:$WINDOW" 2>/dev/null || true
                    fi
                fi
            fi
        fi
    done

    sleep 15  # Check every 15 seconds
done
