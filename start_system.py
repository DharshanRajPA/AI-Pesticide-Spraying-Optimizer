#!/usr/bin/env python3
"""
System startup script for AgriSprayAI.
Starts both API server and React UI in the correct order.
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path

class SystemStarter:
    def __init__(self):
        self.api_process = None
        self.ui_process = None
        self.running = True
        
    def start_api_server(self):
        """Start the API server."""
        print("🚀 Starting API server...")
        try:
            # Activate virtual environment and start server
            if os.name == 'nt':  # Windows
                cmd = ['venv\\Scripts\\python.exe', 'code/api/server.py']
            else:  # Unix/Linux/Mac
                cmd = ['venv/bin/python', 'code/api/server.py']
            
            self.api_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor API server output
            def monitor_api():
                for line in iter(self.api_process.stdout.readline, ''):
                    if self.running:
                        print(f"[API] {line.rstrip()}")
                    else:
                        break
            
            api_thread = threading.Thread(target=monitor_api, daemon=True)
            api_thread.start()
            
            # Wait for server to start
            time.sleep(5)
            
            if self.api_process.poll() is None:
                print("✅ API server started successfully on http://localhost:8000")
                return True
            else:
                print("❌ API server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting API server: {e}")
            return False
    
    def start_react_ui(self):
        """Start the React UI."""
        print("🌐 Starting React UI...")
        try:
            # Change to UI directory and start React
            ui_dir = Path("ui")
            if not ui_dir.exists():
                print("❌ UI directory not found")
                return False
            
            # Check if node_modules exists
            if not (ui_dir / "node_modules").exists():
                print("📦 Installing React dependencies...")
                if os.name == 'nt':  # Windows
                    install_process = subprocess.run(
                        ['npm.cmd', 'install'],
                        cwd=ui_dir,
                        capture_output=True,
                        text=True
                    )
                else:  # Unix/Linux/Mac
                    install_process = subprocess.run(
                        ['npm', 'install'],
                        cwd=ui_dir,
                        capture_output=True,
                        text=True
                    )
                if install_process.returncode != 0:
                    print(f"❌ Failed to install dependencies: {install_process.stderr}")
                    return False
                print("✅ Dependencies installed")
            
            # Start React development server
            if os.name == 'nt':  # Windows
                self.ui_process = subprocess.Popen(
                    ['npm.cmd', 'start'],
                    cwd=ui_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
            else:  # Unix/Linux/Mac
                self.ui_process = subprocess.Popen(
                    ['npm', 'start'],
                    cwd=ui_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
            
            # Monitor UI output
            def monitor_ui():
                for line in iter(self.ui_process.stdout.readline, ''):
                    if self.running:
                        print(f"[UI] {line.rstrip()}")
                    else:
                        break
            
            ui_thread = threading.Thread(target=monitor_ui, daemon=True)
            ui_thread.start()
            
            # Wait for UI to start
            time.sleep(10)
            
            if self.ui_process.poll() is None:
                print("✅ React UI started successfully on http://localhost:3000")
                return True
            else:
                print("❌ React UI failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting React UI: {e}")
            return False
    
    def stop_system(self):
        """Stop all running processes."""
        print("\n🛑 Stopping system...")
        self.running = False
        
        if self.ui_process and self.ui_process.poll() is None:
            print("Stopping React UI...")
            self.ui_process.terminate()
            try:
                self.ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ui_process.kill()
        
        if self.api_process and self.api_process.poll() is None:
            print("Stopping API server...")
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.api_process.kill()
        
        print("✅ System stopped")
    
    def run(self):
        """Main run method."""
        print("🌱 AgriSprayAI System Starter")
        print("=" * 40)
        
        # Check if we're in the right directory
        if not Path("code/api/server.py").exists():
            print("❌ Please run this script from the project root directory")
            return
        
        # Check if virtual environment exists
        venv_path = Path("venv")
        if not venv_path.exists():
            print("❌ Virtual environment not found")
            print("💡 Please run: python -m venv venv")
            return
        
        print("✅ Found project files and virtual environment")
        
        try:
            # Start API server
            if not self.start_api_server():
                print("❌ Failed to start API server")
                return
            
            # Start React UI
            if not self.start_react_ui():
                print("❌ Failed to start React UI")
                return
            
            print("\n🎉 System started successfully!")
            print("\n📱 Access Points:")
            print("   • Web Interface: http://localhost:3000")
            print("   • API Documentation: http://localhost:8000/docs")
            print("   • API Health: http://localhost:8000/health")
            print("\n⏹️  Press Ctrl+C to stop the system")
            
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n⏹️  Shutdown requested by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.stop_system()

def main():
    """Main function."""
    starter = SystemStarter()
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        starter.stop_system()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    starter.run()

if __name__ == "__main__":
    main()
