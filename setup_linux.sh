#!/bin/bash

# Quick Setup Script for Linux
# This script installs all dependencies and prepares the application

echo "==================================="
echo "GPO Fishing Macro - Linux Setup"
echo "==================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Install user packages
echo "Installing user-level packages..."
pip3 install pillow opencv-python-headless numpy pynput mss pyautogui --break-system-packages --user
echo ""

# Install keyboard with sudo
echo "Installing keyboard module (requires root)..."
sudo pip3 install keyboard --break-system-packages
echo ""

# Make the runner script executable
if [ -f "run_linux.sh" ]; then
    chmod +x run_linux.sh
    echo "Made run_linux.sh executable"
else
    echo "Warning: run_linux.sh not found"
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "To run the application, use:"
echo "  ./run_linux.sh"
echo ""
echo "Or manually with:"
echo "  sudo -E python3 GPOfishmacro8.py"
echo ""

