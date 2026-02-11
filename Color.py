import mss
import numpy as np
import keyboard
import sys
import time
from pynput import mouse

def get_cursor_position():
    """Get current mouse cursor position"""
    controller = mouse.Controller()
    return controller.position

def get_color_at_position(x, y):
    """Get RGB color at specific screen position"""
    with mss.mss() as sct:
        # Capture 1x1 pixel at the position
        monitor = {"top": y, "left": x, "width": 1, "height": 1}
        screenshot = sct.grab(monitor)
        
        # Convert to numpy array and get RGB values
        img = np.array(screenshot)
        # mss returns BGRA, convert to RGB
        b, g, r = img[0, 0, :3]
        return (r, g, b)

def main():
    print("Color Picker - Press ESC to exit")
    print("-" * 50)
    
    try:
        while True:
            # Check if ESC is pressed
            if keyboard.is_pressed('esc'):
                print("\nExiting...")
                sys.exit(0)
            
            # Get cursor position
            x, y = get_cursor_position()
            
            # Get color at cursor
            r, g, b = get_color_at_position(x, y)
            
            # Print on same line (overwrite)
            print(f"\rPosition: ({x:4d}, {y:4d}) | RGB: ({r:3d}, {g:3d}, {b:3d}) | HEX: #{r:02X}{g:02X}{b:02X}", end='', flush=True)
            
            # Small delay to reduce CPU usage
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
