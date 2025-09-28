#!/bin/bash
# Webcam Monitor Status Checker

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/webcam_monitor.pid"
LOG_FILE="$SCRIPT_DIR/webcam_clf.log"

echo "=== Webcam Monitor Status ==="

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "Status: NOT RUNNING (no PID file)"
    echo ""
    exit 0
fi

# Read PID
PID=$(cat "$PID_FILE")

# Check if process is running
if kill -0 "$PID" 2>/dev/null; then
    echo "Status: RUNNING"
    echo "PID: $PID"
    echo "Started: $(ps -o lstart= -p "$PID" 2>/dev/null || echo 'unknown')"
    echo ""

    # Show recent log entries
    if [ -f "$LOG_FILE" ]; then
        echo "Recent log entries (last 5 lines):"
        echo "---"
        tail -5 "$LOG_FILE" 2>/dev/null || echo "Unable to read log file"
        echo ""
    fi

    # Show process info
    echo "Process info:"
    ps -p "$PID" -o pid,ppid,rss,vsz,pcpu,pmem,comm 2>/dev/null || echo "Unable to get process info"

else
    echo "Status: NOT RUNNING (stale PID file)"
    echo "Stale PID: $PID"
    echo ""
    echo "Run ./run.sh to start the service"
fi