#!/usr/bin/env python3
"""
Simple AgriSprayAI System Test (No Unicode)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_environment():
    """Test if environment is properly set up."""
    print("Testing Environment Setup...")
    
    # Check if we're in the right directory
    if not Path("code/api/server.py").exists():
        print("FAIL - Not in project directory")
        return False
    
    # Check virtual environment
    if not Path("venv/Scripts/activate.bat").exists():
        print("FAIL - Virtual environment not found")
        return False
    
    # Check .env file
    if not Path(".env").exists():
        print("FAIL - .env file not found")
        return False
    
    print("PASS - Environment setup looks good")
    return True

def test_dependencies():
    """Test if all dependencies are installed."""
    print("Testing Dependencies...")
    
    try:
        # Test Python packages
        import torch
        import ultralytics
        import google.generativeai
        import whisper
        import sentence_transformers
        import fastapi
        import uvicorn
        print("PASS - All Python packages available")
        return True
    except ImportError as e:
        print(f"FAIL - Missing package: {e}")
        return False

def test_models():
    """Test if models are available."""
    print("Testing Models...")
    
    models_to_check = [
        "models/yolov8_baseline/weights/best.pt",
        "models/fusion_model.pt", 
        "models/segmentation_best.pt"
    ]
    
    all_present = True
    for model_path in models_to_check:
        if Path(model_path).exists():
            print(f"PASS - {model_path}")
        else:
            print(f"FAIL - {model_path} missing")
            all_present = False
    
    return all_present

def test_ui_setup():
    """Test if UI is properly set up."""
    print("Testing UI Setup...")
    
    ui_dir = Path("ui")
    if not ui_dir.exists():
        print("FAIL - UI directory not found")
        return False
    
    if not (ui_dir / "package.json").exists():
        print("FAIL - package.json not found in UI directory")
        return False
    
    if not (ui_dir / "node_modules").exists():
        print("FAIL - node_modules not found - need to run npm install")
        return False
    
    print("PASS - UI setup looks good")
    return True

def main():
    """Main test function."""
    print("=" * 50)
    print("AgriSprayAI System Test")
    print("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("Dependencies", test_dependencies), 
        ("Models", test_models),
        ("UI Setup", test_ui_setup)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} Test ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"FAIL - {test_name} test failed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS - All tests passed! Your system is ready!")
        print("\nTo start the system:")
        print("1. Double-click EASY_START.bat (for API server)")
        print("2. Double-click EASY_START_UI.bat (for web interface)")
        print("3. Open browser to http://localhost:3000")
    else:
        print(f"\nWARNING - {total - passed} tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Run: python download_models.py")
        print("- Run: cd ui && npm install")
        print("- Check your .env file has real API keys")

if __name__ == "__main__":
    main()
