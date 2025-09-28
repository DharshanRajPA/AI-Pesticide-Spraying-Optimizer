#!/usr/bin/env python3
"""
System Status Checker for AgriSprayAI
Verifies that all components are working correctly.
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
        print(f"✅ {service_name} is running on port {port}")
        return True
    except requests.exceptions.RequestException:
        print(f"❌ {service_name} is not running on port {port}")
        return False

def check_api_endpoints():
    """Check if API endpoints are responding."""
    try:
        # Check health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API health check passed")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API health check failed: {e}")
        return False

def check_web_interface():
    """Check if the web interface is accessible."""
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Web interface is accessible")
            return True
        else:
            print(f"❌ Web interface check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Web interface check failed: {e}")
        return False

def check_api_docs():
    """Check if API documentation is accessible."""
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API documentation is accessible")
            return True
        else:
            print(f"❌ API documentation check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API documentation check failed: {e}")
        return False

def check_processes():
    """Check if required processes are running."""
    try:
        # Check for Python processes (API server)
        result = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe'], 
                              capture_output=True, text=True)
        if 'python.exe' in result.stdout:
            print("✅ Python processes are running")
            return True
        else:
            print("❌ No Python processes found")
            return False
    except Exception as e:
        print(f"❌ Process check failed: {e}")
        return False

def main():
    """Main status check function."""
    print("🔍 AgriSprayAI System Status Check")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("code/api/server.py").exists():
        print("❌ Please run this script from the project root directory")
        return
    
    print("✅ Found project files")
    print()
    
    # Check services
    print("🌐 Checking Services:")
    api_running = check_port(8000, "API Server")
    web_running = check_port(3000, "Web Interface")
    
    print()
    print("🔗 Checking Endpoints:")
    api_healthy = check_api_endpoints()
    web_accessible = check_web_interface()
    docs_accessible = check_api_docs()
    
    print()
    print("⚙️  Checking Processes:")
    processes_running = check_processes()
    
    print()
    print("📊 Status Summary:")
    print("-" * 20)
    
    total_checks = 6
    passed_checks = sum([
        api_running,
        web_running, 
        api_healthy,
        web_accessible,
        docs_accessible,
        processes_running
    ])
    
    print(f"✅ Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("\n🚀 Your AgriSprayAI system is fully working!")
        print("\n📱 Access Points:")
        print("   • Web Interface: http://localhost:3000")
        print("   • API Documentation: http://localhost:8000/docs")
        print("   • API Health: http://localhost:8000/health")
        
    elif passed_checks >= 4:
        print("\n⚠️  MOSTLY WORKING - Minor issues detected")
        print("\n💡 Try restarting the services:")
        print("   python start_project.py")
        
    else:
        print("\n❌ SYSTEM NOT READY - Multiple issues detected")
        print("\n🛠️  Troubleshooting steps:")
        print("   1. Check if services are running")
        print("   2. Restart with: python start_project.py")
        print("   3. Check the troubleshooting guide")
    
    print("\n📚 For help, see:")
    print("   • TROUBLESHOOTING_GUIDE.md")
    print("   • BEGINNER_SETUP_GUIDE.md")
    print("   • FINAL_WORKING_SETUP.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Status check cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during status check: {e}")
