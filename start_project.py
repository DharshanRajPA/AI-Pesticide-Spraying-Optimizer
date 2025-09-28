#!/usr/bin/env python3
"""
Simple startup script for AgriSprayAI.
This script helps beginners start the project easily.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is installed")
            return True
        else:
            print("❌ Docker is not installed")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False

def check_docker_running():
    """Check if Docker is running."""
    try:
        result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is running")
            return True
        else:
            print("❌ Docker is not running. Please start Docker Desktop.")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False

def check_env_file():
    """Check if .env file exists and has API keys."""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        print("📝 Creating .env file from template...")
        
        # Copy from env.example
        example_file = Path('env.example')
        if example_file.exists():
            with open(example_file, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your actual API keys!")
            return False
        else:
            print("❌ env.example file not found")
            return False
    
    # Check if API keys are set
    with open(env_file, 'r') as f:
        content = f.read()
    
    if 'your_gemini_api_key_here' in content or 'your_kaggle_api_key_here' in content:
        print("⚠️  Please update .env file with your actual API keys!")
        return False
    
    print("✅ .env file looks good")
    return True

def start_with_docker():
    """Start the project using Docker."""
    print("\n🐳 Starting AgriSprayAI with Docker...")
    print("This may take a few minutes on first run...")
    
    try:
        # Start docker-compose
        result = subprocess.run(['docker-compose', 'up', '--build'], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⏹️  Stopping services...")
        subprocess.run(['docker-compose', 'down'])
        return True
    except Exception as e:
        print(f"❌ Error starting Docker: {e}")
        return False

def start_with_python():
    """Start the project using Python directly."""
    print("\n🐍 Starting AgriSprayAI with Python...")
    
    # Check if virtual environment exists
    venv_path = Path('venv')
    if not venv_path.exists():
        print("❌ Virtual environment not found")
        print("📝 Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'])
        print("✅ Virtual environment created")
        print("⚠️  Please activate it and install requirements:")
        print("   venv\\Scripts\\activate")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ Virtual environment found")
    print("⚠️  Please activate the virtual environment and run:")
    print("   venv\\Scripts\\activate")
    print("   python code/api/server.py")
    return True

def main():
    """Main startup function."""
    print("🌱 AgriSprayAI Startup Script")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path('docker-compose.yml').exists():
        print("❌ Please run this script from the project root directory")
        print("   (The directory containing docker-compose.yml)")
        return
    
    print("✅ Found project files")
    
    # Check environment file
    env_ok = check_env_file()
    
    # Check Docker
    docker_installed = check_docker()
    docker_running = False
    
    if docker_installed:
        docker_running = check_docker_running()
    
    print("\n🚀 Starting Options:")
    print("1. Docker (Recommended for beginners)")
    print("2. Python (Advanced users)")
    print("3. Exit")
    
    while True:
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == '1':
            if not docker_installed:
                print("❌ Docker is not installed. Please install Docker Desktop first.")
                print("   Download from: https://www.docker.com/products/docker-desktop/")
                continue
            
            if not docker_running:
                print("❌ Docker is not running. Please start Docker Desktop first.")
                continue
            
            if not env_ok:
                print("❌ Please update your .env file with actual API keys first.")
                continue
            
            start_with_docker()
            break
            
        elif choice == '2':
            start_with_python()
            break
            
        elif choice == '3':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
