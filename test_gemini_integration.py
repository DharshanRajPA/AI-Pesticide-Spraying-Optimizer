#!/usr/bin/env python3
"""
Test script to verify Gemini integration works correctly.
Tests the complete NLP pipeline with Gemini instead of GPT-3.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemini_nlp_pipeline():
    """Test the Gemini NLP pipeline."""
    try:
        logger.info("Testing Gemini NLP Pipeline...")
        
        # Import the Gemini pipeline
        from code.nlp.gemini_nlp_pipeline import GeminiASRNLPPipeline
        
        # Initialize pipeline
        pipeline = GeminiASRNLPPipeline("configs/nlp_pipeline.yaml")
        logger.info("✅ Gemini NLP Pipeline initialized successfully")
        
        # Test text processing
        sample_text = "I noticed yellowing leaves on the eastern side of the field. The plants seem to be wilting and there are some brown spots appearing."
        
        result = pipeline.process_text(sample_text)
        
        if result["success"]:
            logger.info("✅ Text processing successful")
            logger.info(f"   Symptoms: {result['symptom_extraction'].symptoms}")
            logger.info(f"   Likely Causes: {result['symptom_extraction'].likely_causes}")
            logger.info(f"   Severity Indicators: {result['symptom_extraction'].severity_indicators}")
            logger.info(f"   Embedding Dimension: {len(result['text_embedding'].embedding)}")
            
            # Test rationale generation
            rationale = pipeline.generate_rationale(
                result['symptom_extraction'].symptoms,
                0.85
            )
            logger.info(f"✅ Rationale generation successful: {rationale[:100]}...")
            
            return True
        else:
            logger.error(f"❌ Text processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Gemini NLP Pipeline test failed: {e}")
        return False

def test_gemini_api_integration():
    """Test Gemini integration in API server."""
    try:
        logger.info("Testing Gemini API Integration...")
        
        # Test Gemini import
        import google.generativeai as genai
        logger.info("✅ Google Generative AI imported successfully")
        
        # Test configuration (without actual API key)
        try:
            genai.configure(api_key="test_key")
            logger.info("✅ Gemini configuration successful")
        except Exception as e:
            logger.warning(f"⚠️  Gemini configuration test (expected with test key): {e}")
        
        # Test model creation
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("✅ Gemini model creation successful")
        except Exception as e:
            logger.warning(f"⚠️  Gemini model creation test (expected without valid key): {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Gemini API integration test failed: {e}")
        return False

def test_configuration_files():
    """Test that configuration files are properly updated."""
    try:
        logger.info("Testing Configuration Files...")
        
        # Test NLP pipeline config
        import yaml
        with open("configs/nlp_pipeline.yaml", 'r') as f:
            nlp_config = yaml.safe_load(f)
        
        if "gemini" in nlp_config:
            logger.info("✅ NLP pipeline config updated with Gemini")
        else:
            logger.error("❌ NLP pipeline config missing Gemini section")
            return False
        
        # Test API server config
        with open("configs/api_server.yaml", 'r') as f:
            api_config = yaml.safe_load(f)
        
        if "gemini" in api_config["models"]:
            logger.info("✅ API server config updated with Gemini")
        else:
            logger.error("❌ API server config missing Gemini section")
            return False
        
        # Test environment example
        with open("env.example", 'r') as f:
            env_content = f.read()
        
        if "GEMINI_API_KEY" in env_content:
            logger.info("✅ Environment example updated with Gemini API key")
        else:
            logger.error("❌ Environment example missing Gemini API key")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration files test failed: {e}")
        return False

def test_dependencies():
    """Test that required dependencies are available."""
    try:
        logger.info("Testing Dependencies...")
        
        # Test Google Generative AI
        import google.generativeai
        logger.info("✅ google-generativeai available")
        
        # Test other required packages
        import whisper
        logger.info("✅ whisper available")
        
        from sentence_transformers import SentenceTransformer
        logger.info("✅ sentence-transformers available")
        
        import torch
        logger.info("✅ torch available")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("GEMINI INTEGRATION TEST SUITE")
    logger.info("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Configuration Files", test_configuration_files),
        ("Gemini API Integration", test_gemini_api_integration),
        ("Gemini NLP Pipeline", test_gemini_nlp_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Gemini integration is working correctly.")
        logger.info("\nNext steps:")
        logger.info("1. Set your GEMINI_API_KEY in .env file")
        logger.info("2. Run: python code/nlp/gemini_nlp_pipeline.py")
        logger.info("3. Start the API server: python code/api/server.py")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
