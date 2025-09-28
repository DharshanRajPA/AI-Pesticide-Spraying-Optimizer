#!/usr/bin/env python3
"""
Simple server test to verify everything works.
"""

import requests
import time
import subprocess
import sys
from pathlib import Path

def test_server_startup():
    """Test if server can start without errors."""
    print("Testing server startup...")
    
    try:
        # Start server in background
        process = subprocess.Popen(
            [sys.executable, "code/api/server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for startup
        time.sleep(5)
        
        # Check if still running
        if process.poll() is None:
            print("OK - Server started successfully")
            
            # Test health endpoint
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("OK - Health check passed")
                    data = response.json()
                    print(f"   Models loaded: {data.get('models_loaded', {})}")
                else:
                    print(f"FAIL - Health check failed: {response.status_code}")
            except Exception as e:
                print(f"FAIL - Health check error: {e}")
            
            # Stop server
            process.terminate()
            process.wait()
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"FAIL - Server failed to start")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"FAIL - Server test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 50)
    print("AgriSprayAI Server Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("code/api/server.py").exists():
        print("FAIL - Not in project directory")
        return
    
    # Test server
    success = test_server_startup()
    
    if success:
        print("\nSUCCESS - Server test passed! Your system is ready!")
        print("\nTo start the system:")
        print("1. Double-click SIMPLE_START.bat (for API server)")
        print("2. Double-click EASY_START_UI.bat (for web interface)")
        print("3. Open browser to http://localhost:3000")
    else:
        print("\nFAIL - Server test failed. Please check the errors above.")

if __name__ == "__main__":
    main()
