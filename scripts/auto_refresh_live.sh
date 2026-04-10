#!/bin/bash
# Auto-refresh live predictions every 5 minutes and push to GitHub.
# Run: bash scripts/auto_refresh_live.sh
# Stop: kill $(cat data/live/auto_refresh.pid)

REPO="/Users/dylanjaynes/Augusta National Model"
INTERVAL=300  # seconds between refreshes
LOG="$REPO/data/live/auto_refresh.log"

mkdir -p "$REPO/data/live"
echo $$ > "$REPO/data/live/auto_refresh.pid"

echo "[$(date '+%H:%M:%S')] Auto-refresh started (PID $$, every ${INTERVAL}s)" | tee -a "$LOG"

while true; do
    TIMESTAMP=$(date '+%H:%M:%S')

    # Run live inference
    cd "$REPO"
    python3 scripts/run_live_inference.py --dg-live >> "$LOG" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        # Commit and push the updated CSV
        git add data/live/live_predictions_latest.csv >> "$LOG" 2>&1
        CHANGES=$(git diff --cached --name-only)
        if [ -n "$CHANGES" ]; then
            git commit -m "live: auto-refresh $(date '+%Y-%m-%d %H:%M')" >> "$LOG" 2>&1
            git push origin main >> "$LOG" 2>&1
            echo "[$TIMESTAMP] Pushed fresh predictions" | tee -a "$LOG"
        else
            echo "[$TIMESTAMP] No changes to push (data unchanged)" | tee -a "$LOG"
        fi
    else
        echo "[$TIMESTAMP] Inference failed (exit $EXIT_CODE) — skipping push" | tee -a "$LOG"
    fi

    echo "[$TIMESTAMP] Sleeping ${INTERVAL}s..." | tee -a "$LOG"
    sleep $INTERVAL
done
