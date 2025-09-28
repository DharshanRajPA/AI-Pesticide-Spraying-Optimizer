#!/usr/bin/env python3
"""
Simple startup script for AgriSprayAI
"""

import uvicorn
import os
import sys

def main():
    print("🌾 AgriSprayAI - Pest Detection & Spraying Optimization")
    print("=" * 60)
    print("Starting server...")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("🔍 Upload an image to detect pests and get spraying recommendations")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down AgriSprayAI...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
