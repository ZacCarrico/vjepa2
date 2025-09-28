#!/bin/bash
# Webcam Monitor Runner
# Starts the webcam monitor service in the background

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/webcam_clf.py"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
PID_FILE="$SCRIPT_DIR/webcam_monitor.pid"
LOG_FILE="$SCRIPT_DIR/webcam_clf.log"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Webcam monitor is already running (PID: $PID)"
        exit 1
    else
        echo "Removing stale PID file"
        rm "$PID_FILE"
    fi
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Make sure Python script is executable
chmod +x "$PYTHON_SCRIPT"

echo "Starting webcam monitor..."
echo "Script: $PYTHON_SCRIPT"
echo "Config: $CONFIG_FILE"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"

# Start the monitor in the background using nohup
nohup python3 "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &

# Save the PID
echo $! > "$PID_FILE"

echo "Webcam monitor started with PID: $(cat $PID_FILE)"
echo "Logs will be written to: $LOG_FILE"
echo ""
echo "To stop the monitor, run: ./stop.sh"
echo "To view logs, run: tail -f $LOG_FILE"