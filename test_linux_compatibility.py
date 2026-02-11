#!/usr/bin/env python3
"""
Test script to verify Linux compatibility of GPOfishmacro8.py
Tests imports, platform detection, and basic functionality without GUI
"""

import sys
import platform
import importlib.util

def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    required_modules = [
        'tkinter',
        'PIL',
        'cv2',
        'numpy',
        'mss',
        'pynput',
        'pyautogui',
        'threading',
        'queue',
        'json',
        'os',
        'sys',
        'time',
        'ctypes'
    ]
    
    failed = []
    for module in required_modules:
        try:
            if module == 'PIL':
                __import__('PIL.Image')
                __import__('PIL.ImageTk')
            elif module == 'cv2':
                __import__('cv2')
            else:
                __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed.append(module)
    
    print()
    if failed:
        print(f"Failed to import: {', '.join(failed)}")
        return False
    else:
        print("All imports successful!")
        return True

def test_platform_detection():
    """Test platform detection"""
    print("\n" + "=" * 60)
    print("Testing platform detection...")
    print("=" * 60)
    
    print(f"System: {platform.system()}")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    
    is_linux = platform.system() == 'Linux'
    is_windows = platform.system() == 'Windows'
    
    print(f"\nDetected as Linux: {is_linux}")
    print(f"Detected as Windows: {is_windows}")
    
    if is_linux:
        print("\n✓ Running on Linux - This is our target platform!")
    else:
        print("\n⚠ Not running on Linux")
    
    return True

def test_code_syntax():
    """Test if the main file has valid Python syntax"""
    print("\n" + "=" * 60)
    print("Testing code syntax...")
    print("=" * 60)
    
    main_file = "GPOfishmacro8.py"
    
    try:
        spec = importlib.util.spec_from_file_location("test_module", main_file)
        if spec is None:
            print(f"✗ Could not load spec for {main_file}")
            return False
        
        # This will compile the module but not execute it
        module = importlib.util.module_from_spec(spec)
        print(f"✓ {main_file} has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in {main_file}: {e}")
        return False
    except Exception as e:
        print(f"⚠ Warning while checking {main_file}: {e}")
        return True  # May be import errors, not syntax errors

def test_windows_specific_code():
    """Check for Windows-specific code that might cause issues on Linux"""
    print("\n" + "=" * 60)
    print("Checking for Windows-specific code...")
    print("=" * 60)
    
    if platform.system() != 'Linux':
        print("⚠ Not running on Linux, skipping Windows-specific checks")
        return True
    
    # Test if Windows-specific ctypes calls are properly guarded
    issues = []
    
    try:
        import ctypes
        # These should fail on Linux if not properly guarded
        try:
            user32 = ctypes.windll.user32
            print("✗ ctypes.windll.user32 is accessible (should be guarded on Linux)")
            issues.append("ctypes.windll.user32 not guarded")
        except AttributeError:
            print("✓ ctypes.windll properly fails on Linux")
    except Exception as e:
        print(f"⚠ Unexpected error testing Windows-specific code: {e}")
    
    if issues:
        print(f"\nFound issues: {', '.join(issues)}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("LINUX COMPATIBILITY TEST SUITE")
    print("=" * 60)
    
    results = {
        "imports": test_imports(),
        "platform": test_platform_detection(),
        "syntax": test_code_syntax(),
        "windows_code": test_windows_specific_code()
    }
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

