#!/usr/bin/env python3
"""
ASR and NLP pipeline for AgriSprayAI using Google Gemini.
Implements Whisper for speech recognition and Gemini for structured symptom extraction.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import asyncio
import tempfile
import wave
import struct

# Audio processing
import whisper
import torch
import torchaudio
import librosa
import numpy as np

# NLP and text processing
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import re
from dataclasses import dataclass
from enum import Enum

# Configuration
import yaml
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"

@dataclass
class TranscriptionResult:
    """Result of audio transcription."""
    text: str
    confidence: float
    language: str
    segments: List[Dict[str, Any]]
    processing_time: float

@dataclass
class SymptomExtraction:
    """Structured symptom extraction result."""
    symptoms: List[str]
    likely_causes: List[str]
    severity_indicators: List[str]
    clarifying_questions: List[str]
    confidence: float
    processing_time: float

@dataclass
class TextEmbedding:
    """Text embedding result."""
    embedding: List[float]
    model_name: str
    processing_time: float

class WhisperASR:
    """Whisper-based Automatic Speech Recognition."""
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.device = self._get_device(device)
        self.model = None
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"Whisper model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> TranscriptionResult:
        """Transcribe audio file to text."""
        import time
        start_time = time.time()
        
        try:
            # Load and preprocess audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Transcribe
            result = self.model.transcribe(
                audio,
                language=language,
                word_timestamps=True,
                verbose=False
            )
            
            processing_time = time.time() - start_time
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "confidence": segment.get("avg_logprob", 0.0)
                })
            
            return TranscriptionResult(
                text=result["text"].strip(),
                confidence=result.get("language_probability", 0.0),
                language=result.get("language", "unknown"),
                segments=segments,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_bytes(self, audio_bytes: bytes, format: AudioFormat = AudioFormat.WAV) -> TranscriptionResult:
        """Transcribe audio from bytes."""
        with tempfile.NamedTemporaryFile(suffix=f".{format.value}", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            
            try:
                result = self.transcribe_audio(temp_file.name)
                return result
            finally:
                os.unlink(temp_file.name)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(whisper.tokenizer.LANGUAGES.keys())

class GeminiSymptomExtractor:
    """Google Gemini based structured symptom extraction."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash", temperature: float = 0.1):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        
        # Define the system prompt for structured extraction
        self.system_prompt = """You are an agricultural expert specializing in pest and disease identification. 
        Your task is to extract structured information from farmer notes about crop conditions.
        
        Respond with a JSON object containing:
        - "symptoms": List of specific symptoms mentioned (e.g., "yellowing leaves", "brown spots")
        - "likely_causes": List of potential pest/disease causes (e.g., "aphid infestation", "fungal infection")
        - "severity_indicators": List of words/phrases indicating severity (e.g., "severe", "widespread", "mild")
        - "clarifying_questions": List of questions to ask for more information
        
        Be precise and focus on agricultural terminology. If no relevant information is found, return empty lists.
        """
    
    def extract_symptoms(self, text: str) -> SymptomExtraction:
        """Extract structured symptoms from text."""
        import time
        start_time = time.time()
        
        try:
            # Create the prompt
            prompt = f"{self.system_prompt}\n\nExtract symptoms from this farmer note: {text}"
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=500,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            # Parse JSON response
            content = response.text.strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                extracted_data = json.loads(json_str)
            else:
                # Fallback parsing
                extracted_data = self._fallback_parsing(content)
            
            processing_time = time.time() - start_time
            
            return SymptomExtraction(
                symptoms=extracted_data.get("symptoms", []),
                likely_causes=extracted_data.get("likely_causes", []),
                severity_indicators=extracted_data.get("severity_indicators", []),
                clarifying_questions=extracted_data.get("clarifying_questions", []),
                confidence=0.8,  # Placeholder confidence
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Symptom extraction failed: {e}")
            # Return empty result on failure
            return SymptomExtraction(
                symptoms=[],
                likely_causes=[],
                severity_indicators=[],
                clarifying_questions=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def _fallback_parsing(self, content: str) -> Dict[str, List[str]]:
        """Fallback parsing when JSON extraction fails."""
        # Simple keyword-based extraction
        symptoms = []
        causes = []
        severity_indicators = []
        questions = []
        
        # Common symptom keywords
        symptom_keywords = [
            "yellowing", "browning", "wilting", "spots", "holes", "discoloration",
            "stunted", "deformed", "drooping", "curling", "necrosis"
        ]
        
        # Common cause keywords
        cause_keywords = [
            "aphid", "beetle", "worm", "mite", "fungus", "bacteria", "virus",
            "infection", "infestation", "disease", "pest"
        ]
        
        # Severity indicators
        severity_keywords = [
            "severe", "mild", "moderate", "widespread", "localized", "extensive",
            "heavy", "light", "significant", "minor"
        ]
        
        text_lower = content.lower()
        
        for keyword in symptom_keywords:
            if keyword in text_lower:
                symptoms.append(keyword)
        
        for keyword in cause_keywords:
            if keyword in text_lower:
                causes.append(keyword)
        
        for keyword in severity_keywords:
            if keyword in text_lower:
                severity_indicators.append(keyword)
        
        return {
            "symptoms": symptoms,
            "likely_causes": causes,
            "severity_indicators": severity_indicators,
            "clarifying_questions": questions
        }
    
    def generate_rationale(self, symptoms: List[str], vision_confidence: float) -> str:
        """Generate human-readable rationale for the analysis."""
        try:
            prompt = f"""Based on the following symptoms: {', '.join(symptoms)}
            And vision model confidence: {vision_confidence:.2f}
            
            Provide a brief 2-3 sentence explanation of the pest/disease assessment and recommended action.
            Keep it simple and actionable for farmers."""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=150,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Rationale generation failed: {e}")
            return "Analysis completed. Please review the detected symptoms and confidence scores."

class TextEmbedder:
    """Text embedding using sentence transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load sentence transformer model."""
        try:
            logger.info(f"Loading text embedder: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Text embedder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text embedder: {e}")
            raise
    
    def embed_text(self, text: str) -> TextEmbedding:
        """Generate text embedding."""
        import time
        start_time = time.time()
        
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            processing_time = time.time() - start_time
            
            return TextEmbedding(
                embedding=embedding.tolist(),
                model_name=self.model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[TextEmbedding]:
        """Generate embeddings for multiple texts."""
        import time
        start_time = time.time()
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            processing_time = time.time() - start_time
            
            results = []
            for i, embedding in enumerate(embeddings):
                results.append(TextEmbedding(
                    embedding=embedding.tolist(),
                    model_name=self.model_name,
                    processing_time=processing_time / len(texts)
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch text embedding failed: {e}")
            raise

class GeminiASRNLPPipeline:
    """Complete ASR and NLP pipeline for AgriSprayAI using Gemini."""
    
    def __init__(self, config_path: str = "configs/nlp_pipeline.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.whisper = WhisperASR(
            model_size=self.config["whisper"]["model_size"],
            device=self.config["whisper"]["device"]
        )
        
        self.symptom_extractor = GeminiSymptomExtractor(
            model=self.config["gemini"]["model"],
            temperature=self.config["gemini"]["temperature"]
        )
        
        self.text_embedder = TextEmbedder(
            model_name=self.config["text_embedder"]["model_name"]
        )
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the pipeline."""
        log_dir = Path(self.config["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def process_audio(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """Process audio file through the complete pipeline."""
        try:
            # Step 1: Transcribe audio
            transcription = self.whisper.transcribe_audio(audio_path, language)
            
            # Step 2: Extract symptoms if text is available
            symptom_extraction = None
            if transcription.text.strip():
                symptom_extraction = self.symptom_extractor.extract_symptoms(transcription.text)
            
            # Step 3: Generate text embedding
            text_embedding = None
            if transcription.text.strip():
                text_embedding = self.text_embedder.embed_text(transcription.text)
            
            return {
                "transcription": transcription,
                "symptom_extraction": symptom_extraction,
                "text_embedding": text_embedding,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return {
                "transcription": None,
                "symptom_extraction": None,
                "text_embedding": None,
                "success": False,
                "error": str(e)
            }
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text through the NLP pipeline."""
        try:
            # Step 1: Extract symptoms
            symptom_extraction = self.symptom_extractor.extract_symptoms(text)
            
            # Step 2: Generate text embedding
            text_embedding = self.text_embedder.embed_text(text)
            
            return {
                "symptom_extraction": symptom_extraction,
                "text_embedding": text_embedding,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return {
                "symptom_extraction": None,
                "text_embedding": None,
                "success": False,
                "error": str(e)
            }
    
    def generate_rationale(self, symptoms: List[str], vision_confidence: float) -> str:
        """Generate human-readable rationale."""
        return self.symptom_extractor.generate_rationale(symptoms, vision_confidence)
    
    def validate_against_vision(self, text_confidence: float, vision_confidence: float) -> Dict[str, Any]:
        """Validate text analysis against vision model confidence."""
        confidence_diff = abs(text_confidence - vision_confidence)
        
        if confidence_diff > 0.3:
            return {
                "validation": "warning",
                "message": "Significant difference between text and vision confidence",
                "confidence_diff": confidence_diff,
                "requires_review": True
            }
        elif confidence_diff > 0.2:
            return {
                "validation": "caution",
                "message": "Moderate difference between text and vision confidence",
                "confidence_diff": confidence_diff,
                "requires_review": False
            }
        else:
            return {
                "validation": "good",
                "message": "Text and vision confidence are aligned",
                "confidence_diff": confidence_diff,
                "requires_review": False
            }

def main():
    """Example usage of the Gemini ASR/NLP pipeline."""
    # Initialize pipeline
    pipeline = GeminiASRNLPPipeline()
    
    # Example: Process text
    sample_text = "I noticed yellowing leaves on the eastern side of the field. The plants seem to be wilting and there are some brown spots appearing."
    
    result = pipeline.process_text(sample_text)
    
    if result["success"]:
        print("Text Processing Results:")
        print(f"Symptoms: {result['symptom_extraction'].symptoms}")
        print(f"Likely Causes: {result['symptom_extraction'].likely_causes}")
        print(f"Severity Indicators: {result['symptom_extraction'].severity_indicators}")
        print(f"Embedding Dimension: {len(result['text_embedding'].embedding)}")
        
        # Generate rationale
        rationale = pipeline.generate_rationale(
            result['symptom_extraction'].symptoms,
            0.85
        )
        print(f"Rationale: {rationale}")
    else:
        print(f"Processing failed: {result['error']}")

if __name__ == "__main__":
    main()
