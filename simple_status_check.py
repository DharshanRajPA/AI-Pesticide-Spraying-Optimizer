#!/usr/bin/env python3
"""
Simple System Status Checker for AgriSprayAI
Checks if all services are running without Unicode issues.
"""

import requests
import subprocess
import sys
import time
from pathlib import Path

def check_port(port, service_name):
    """Check if a service is running on a specific port."""
    try:
        response = requests.get(f"http://localhost:{port}", timeout=5)
        print(f"OK - {service_name} is running on port {port}")
        return True
    except requests.exceptions.RequestException:
        print(f"FAIL - {service_name} is not running on port {port}")
        return False

def check_api_endpoints():
    """Check if API endpoints are responding."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("OK - API health check passed")
            return True
        else:
            print(f"FAIL - API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"FAIL - API health check failed: {e}")
        return False

def check_web_interface():
    """Check if the web interface is accessible."""
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("OK - Web interface is accessible")
            return True
        else:
            print(f"FAIL - Web interface check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"FAIL - Web interface check failed: {e}")
        return False

def check_api_docs():
    """Check if API documentation is accessible."""
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("OK - API documentation is accessible")
            return True
        else:
            print(f"FAIL - API documentation check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"FAIL - API documentation check failed: {e}")
        return False

def main():
    """Main status check function."""
    print("AgriSprayAI System Status Check")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("code/api/server.py").exists():
        print("FAIL - Please run this script from the project root directory")
        return
    
    print("OK - Found project files")
    print()
    
    # Check services
    print("Checking Services:")
    api_running = check_port(8000, "API Server")
    web_running = check_port(3000, "Web Interface")
    
    print()
    print("Checking Endpoints:")
    api_healthy = check_api_endpoints()
    web_accessible = check_web_interface()
    docs_accessible = check_api_docs()
    
    print()
    print("Status Summary:")
    print("-" * 20)
    
    total_checks = 5
    passed_checks = sum([
        api_running,
        web_running, 
        api_healthy,
        web_accessible,
        docs_accessible
    ])
    
    print(f"Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("\nSUCCESS - ALL SYSTEMS OPERATIONAL!")
        print("\nYour AgriSprayAI system is fully working!")
        print("\nAccess Points:")
        print("   - Web Interface: http://localhost:3000")
        print("   - API Documentation: http://localhost:8000/docs")
        print("   - API Health: http://localhost:8000/health")
        
    elif passed_checks >= 3:
        print("\nMOSTLY WORKING - Minor issues detected")
        print("\nTry restarting the services:")
        print("   python start_system.py")
        
    else:
        print("\nSYSTEM NOT READY - Multiple issues detected")
        print("\nTroubleshooting steps:")
        print("   1. Check if services are running")
        print("   2. Restart with: python start_system.py")
        print("   3. Check the troubleshooting guide")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStatus check cancelled by user.")
    except Exception as e:
        print(f"\nError during status check: {e}")
