#!/bin/bash

# GPO Fishing Macro - Linux Runner Script
# This script runs the GPO fishing macro with proper permissions on Linux

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script needs root privileges to access keyboard hooks."
    echo "Restarting with sudo..."
    exec sudo -E "$0" "$@"
fi

# Preserve the original user's DISPLAY and XAUTHORITY for GUI
if [ -z "$SUDO_USER" ]; then
    export DISPLAY=${DISPLAY:-:0}
else
    export DISPLAY=:0
    export XAUTHORITY=/home/$SUDO_USER/.Xauthority
fi

# Run the Python script
python3 "$(dirname "$0")/GPOfishmacro8.py"

