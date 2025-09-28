#!/usr/bin/env python3
"""
Simple test to check if the API server can start properly.
"""

import sys
import os
from pathlib import Path

# Add the code directory to the path
sys.path.append(str(Path(__file__).parent / "code"))

def test_imports():
    """Test if all imports work."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import whisper
        print("✅ Whisper imported successfully")
    except ImportError as e:
        print(f"❌ Whisper import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformer imported successfully")
    except ImportError as e:
        print(f"❌ SentenceTransformer import failed: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI imported successfully")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        return False
    
    try:
        from fastapi import FastAPI
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    return True

def test_config():
    """Test if configuration loads properly."""
    print("\n🔍 Testing configuration...")
    
    try:
        import yaml
        with open("configs/api_server.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration load failed: {e}")
        return False

def test_models():
    """Test if models can be loaded."""
    print("\n🔍 Testing model loading...")
    
    try:
        import whisper
        model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Whisper model load failed: {e}")
        return False

def test_server_creation():
    """Test if FastAPI app can be created."""
    print("\n🔍 Testing server creation...")
    
    try:
        from fastapi import FastAPI
        app = FastAPI(title="Test API")
        print("✅ FastAPI app created successfully")
        return True
    except Exception as e:
        print(f"❌ FastAPI app creation failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 AgriSprayAI Server Test")
    print("=" * 30)
    
    tests = [
        test_imports,
        test_config,
        test_models,
        test_server_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Server should work.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
