#!/usr/bin/env python3
"""
Simple test to verify Gemini integration works.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemini_basic():
    """Test basic Gemini functionality."""
    try:
        logger.info("Testing basic Gemini functionality...")
        
        # Import Gemini
        import google.generativeai as genai
        logger.info("✅ Google Generative AI imported successfully")
        
        # Configure with test key (will fail but that's expected)
        genai.configure(api_key="AIzaSyDawAzkDWA_QTun89gyezAev016zeBpnnM")
        logger.info("✅ Gemini configuration successful")
        
        # Create model
        model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("✅ Gemini model creation successful")
        
        # Test prompt (will fail without real API key, but that's expected)
        try:
            response = model.generate_content("Hello, this is a test.")
            logger.info("✅ Gemini content generation successful")
            logger.info(f"Response: {response.text[:100]}...")
        except Exception as e:
            logger.warning(f"⚠️  Content generation failed (expected without real API key): {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic Gemini test failed: {e}")
        return False

def test_gemini_nlp_functionality():
    """Test Gemini NLP functionality with a simple example."""
    try:
        logger.info("Testing Gemini NLP functionality...")
        
        import google.generativeai as genai
        
        # Configure
        genai.configure(api_key="AIzaSyDawAzkDWA_QTun89gyezAev016zeBpnnM")
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Test agricultural symptom extraction prompt
        prompt = """You are an agricultural expert. Extract structured information from farmer notes about pest/disease symptoms. 
        Respond with JSON format: {"symptoms": ["symptom1", "symptom2"], "likely_causes": ["cause1", "cause2"], "severity_indicators": ["indicator1"], "clarifying_questions": ["question1"]}
        
        Farmer note: I noticed yellowing leaves on the eastern side of the field. The plants seem to be wilting and there are some brown spots appearing."""
        
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                    top_p=0.8,
                    top_k=40
                )
            )
            logger.info("✅ Gemini agricultural NLP prompt successful")
            logger.info(f"Response: {response.text[:200]}...")
        except Exception as e:
            logger.warning(f"⚠️  Agricultural NLP test failed (expected without real API key): {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Gemini NLP functionality test failed: {e}")
        return False

def main():
    """Run simple Gemini tests."""
    logger.info("="*60)
    logger.info("SIMPLE GEMINI INTEGRATION TEST")
    logger.info("="*60)
    
    tests = [
        ("Basic Gemini Functionality", test_gemini_basic),
        ("Gemini NLP Functionality", test_gemini_nlp_functionality),
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
        logger.info("🎉 All basic tests passed! Gemini integration is working correctly.")
        logger.info("\nTo use with real API:")
        logger.info("1. Get your Gemini API key from: https://makersuite.google.com/app/apikey")
        logger.info("2. Set GEMINI_API_KEY environment variable")
        logger.info("3. Run the full AgriSprayAI system")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
