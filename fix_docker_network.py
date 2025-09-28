#!/usr/bin/env python3
"""
Docker Network Troubleshooting Script for AgriSprayAI
Helps diagnose and fix Docker networking issues.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_command(command, capture_output=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_docker_status():
    """Check if Docker is running and accessible."""
    print("🔍 Checking Docker status...")
    
    # Check Docker version
    success, stdout, stderr = run_command("docker --version")
    if not success:
        print("❌ Docker is not installed or not in PATH")
        return False
    print(f"✅ Docker version: {stdout.strip()}")
    
    # Check Docker daemon
    success, stdout, stderr = run_command("docker info")
    if not success:
        print("❌ Docker daemon is not running")
        print("💡 Please start Docker Desktop and wait for it to fully load")
        return False
    print("✅ Docker daemon is running")
    
    return True

def check_network_connectivity():
    """Check network connectivity to Docker registries."""
    print("\n🌐 Checking network connectivity...")
    
    # Test basic internet connectivity
    success, stdout, stderr = run_command("ping -n 1 8.8.8.8")
    if not success:
        print("❌ No internet connectivity")
        return False
    print("✅ Internet connectivity OK")
    
    # Test Docker Hub connectivity
    success, stdout, stderr = run_command("ping -n 1 registry-1.docker.io")
    if not success:
        print("⚠️  Cannot reach Docker Hub registry")
        print("💡 This might be a DNS or firewall issue")
    else:
        print("✅ Docker Hub registry reachable")
    
    return True

def test_docker_pull():
    """Test pulling a simple image."""
    print("\n📦 Testing Docker image pull...")
    
    # Try to pull a simple image
    success, stdout, stderr = run_command("docker pull hello-world")
    if not success:
        print("❌ Failed to pull test image")
        print(f"Error: {stderr}")
        return False
    
    print("✅ Successfully pulled test image")
    
    # Clean up
    run_command("docker rmi hello-world")
    print("✅ Cleaned up test image")
    
    return True

def fix_docker_network():
    """Try to fix Docker networking issues."""
    print("\n🔧 Attempting to fix Docker networking...")
    
    # Restart Docker network
    print("🔄 Restarting Docker networks...")
    run_command("docker network prune -f")
    
    # Reset Docker to factory defaults (if needed)
    print("💡 If issues persist, try:")
    print("   1. Restart Docker Desktop")
    print("   2. Reset Docker to factory defaults in Docker Desktop settings")
    print("   3. Check Windows Firewall settings")
    print("   4. Try using a VPN if in a restricted network")

def create_simple_compose():
    """Create a simplified docker-compose file for testing."""
    print("\n📝 Creating simplified docker-compose for testing...")
    
    simple_compose = """services:
  # Test service
  test:
    image: nginx:alpine
    ports:
      - "8080:80"
    restart: unless-stopped

networks:
  default:
    driver: bridge
"""
    
    with open("docker-compose.test.yml", "w") as f:
        f.write(simple_compose)
    
    print("✅ Created docker-compose.test.yml")
    print("💡 You can test with: docker-compose -f docker-compose.test.yml up")

def main():
    """Main troubleshooting function."""
    print("🐳 AgriSprayAI Docker Network Troubleshooter")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("docker-compose.yml").exists():
        print("❌ Please run this script from the project root directory")
        return
    
    print("✅ Found docker-compose.yml")
    
    # Step 1: Check Docker status
    if not check_docker_status():
        print("\n🛑 Docker is not ready. Please fix Docker issues first.")
        return
    
    # Step 2: Check network connectivity
    if not check_network_connectivity():
        print("\n🛑 Network connectivity issues detected.")
        fix_docker_network()
        return
    
    # Step 3: Test Docker pull
    if not test_docker_pull():
        print("\n🛑 Docker image pull failed.")
        fix_docker_network()
        return
    
    # Step 4: Create test compose file
    create_simple_compose()
    
    print("\n🎉 Docker appears to be working correctly!")
    print("\n🚀 Next steps:")
    print("1. Try the simplified compose: docker-compose -f docker-compose.simple.yml up --build")
    print("2. Or test with: docker-compose -f docker-compose.test.yml up")
    print("3. If that works, try the full compose: docker-compose up --build")
    
    print("\n💡 If you still have issues:")
    print("- Check Windows Firewall settings")
    print("- Try using a different network (mobile hotspot)")
    print("- Restart Docker Desktop")
    print("- Reset Docker to factory defaults")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Troubleshooting cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during troubleshooting: {e}")
