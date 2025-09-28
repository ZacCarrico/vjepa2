#!/bin/bash
# Webcam Monitor Stopper
# Stops the webcam monitor service

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/webcam_monitor.pid"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "Webcam monitor is not running (no PID file found)"
    exit 1
fi

# Read PID
PID=$(cat "$PID_FILE")

# Check if process is actually running
if ! kill -0 "$PID" 2>/dev/null; then
    echo "Process $PID is not running, removing stale PID file"
    rm "$PID_FILE"
    exit 1
fi

echo "Stopping webcam monitor (PID: $PID)..."

# Try graceful shutdown first (SIGTERM)
kill "$PID"

# Wait up to 10 seconds for graceful shutdown
for i in {1..10}; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Webcam monitor stopped successfully"
        rm "$PID_FILE"
        exit 0
    fi
    echo "Waiting for graceful shutdown... ($i/10)"
    sleep 1
done

# Force kill if graceful shutdown failed
echo "Forcing shutdown..."
kill -9 "$PID" 2>/dev/null

# Wait a bit more and check
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "Webcam monitor force stopped"
    rm "$PID_FILE"
    exit 0
else
    echo "Error: Unable to stop webcam monitor process $PID"
    exit 1
fi