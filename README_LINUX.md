# GPO Fishing Macro - Linux Setup Instructions

## Prerequisites

This application has been updated to work on both Windows and Linux platforms.

### Required Python Packages

Install the following packages:

```bash
pip3 install keyboard pillow opencv-python-headless numpy pynput mss pyautogui --break-system-packages
```

**Note:** The `keyboard` module requires root privileges on Linux, so you'll need to install it system-wide:

```bash
sudo pip3 install keyboard --break-system-packages
```

## Running on Linux

The `keyboard` module requires root privileges to hook into keyboard events on Linux. You have two options:

### Option 1: Using the Helper Script (Recommended)

1. Make the script executable:
   ```bash
   chmod +x run_linux.sh
   ```

2. Run the script:
   ```bash
   ./run_linux.sh
   ```
   
   The script will automatically request sudo privileges and preserve your display environment for the GUI.

### Option 2: Direct Execution

Run with sudo and preserve the display environment:

```bash
sudo -E python3 GPOfishmacro8.py
```

Or set the display manually:

```bash
sudo DISPLAY=:0 python3 GPOfishmacro8.py
```

## Important Notes

1. **Root Privileges**: The application needs root access on Linux because the `keyboard` module hooks into low-level keyboard events for the macro functionality.

2. **Display Environment**: When running with sudo, you need to preserve the DISPLAY environment variable so the GUI can connect to your X server.

3. **Security**: Be cautious when running applications with root privileges. Review the code if you have concerns.

4. **Virtual Environment Alternative**: Instead of using `--break-system-packages`, you could create a virtual environment, but you'll still need to run the script with sudo:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install keyboard pillow opencv-python-headless numpy pynput mss pyautogui
   deactivate
   sudo venv/bin/python GPOfishmacro8.py
   ```

## Troubleshooting

### "No module named 'keyboard'" with sudo

The `keyboard` module needs to be installed system-wide or in a location accessible to root:

```bash
sudo pip3 install keyboard --break-system-packages
```

### GUI doesn't appear

Make sure the DISPLAY environment variable is set:

```bash
sudo -E python3 GPOfishmacro8.py
```

Or use the provided `run_linux.sh` script which handles this automatically.

### Permission denied errors

Some Linux distributions have stricter security policies. You may need to:
1. Check your system's polkit policies
2. Ensure your user is in the `input` group (though this won't help with the keyboard module)
3. Review your system's security policies regarding input device access

## Platform-Specific Behavior

The application automatically detects your platform and uses appropriate methods:
- **Windows**: Uses win32 APIs for window management and DPI awareness
- **Linux**: Uses X11 through tkinter and other cross-platform libraries

Both versions should provide equivalent functionality.

