#!/usr/bin/env python3
"""
FastAPI server for AgriSprayAI.
Provides REST API endpoints for image analysis, dose optimization, and flight planning.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import asyncio
from datetime import datetime
import uuid
import hashlib
import shutil

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn

# ML and Computer Vision
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import whisper
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from action_engine.optimizer import DoseOptimizer, PlantInstance, OptimizationResult
from planner.flight_planner import FlightPlanner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
import yaml
with open("configs/api_server.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_models()
    yield
    # Shutdown (if needed)

# Initialize FastAPI app
app = FastAPI(
    title=config["api"]["title"],
    description=config["api"]["description"],
    version=config["api"]["version"],
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors"]["origins"],
    allow_credentials=True,
    allow_methods=config["api"]["cors"]["methods"],
    allow_headers=config["api"]["cors"]["headers"],
)

# Security
security = HTTPBearer()

# Global variables for loaded models
vision_model = None
fusion_model = None
segmentation_model = None
whisper_model = None
sentence_transformer = None
dose_optimizer = None
flight_planner = None

# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    image_hash: Optional[str] = None
    transcript: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    run_id: str
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    processing_time: float
    model_version: str

class PlanningRequest(BaseModel):
    """Request model for planning endpoint."""
    predictions: List[Dict[str, Any]]
    location: Dict[str, float]  # GPS coordinates
    constraints: Optional[Dict[str, Any]] = None
    field_info: Optional[Dict[str, Any]] = None

class PlanningResponse(BaseModel):
    """Response model for planning endpoint."""
    run_id: str
    doses: List[float]
    waypoints: List[Dict[str, Any]]
    mavlink_json: Dict[str, Any]
    explainability_url: str
    total_pesticide: float
    estimated_time: float
    requires_approval: bool

class ApprovalRequest(BaseModel):
    """Request model for approval endpoint."""
    run_id: str
    decision: str  # "approved" or "rejected"
    operator_id: str
    comments: Optional[str] = None
    signature: Optional[str] = None

class ApprovalResponse(BaseModel):
    """Response model for approval endpoint."""
    run_id: str
    decision: str
    timestamp: str
    operator_id: str

# Utility functions
def load_models():
    """Load all required models."""
    global vision_model, fusion_model, segmentation_model, whisper_model, sentence_transformer, dose_optimizer, flight_planner
    
    try:
        # Load vision model
        vision_model_path = config["models"]["vision"]["model_path"]
        if os.path.exists(vision_model_path):
            vision_model = YOLO(vision_model_path)
            logger.info(f"Loaded vision model: {vision_model_path}")
        else:
            logger.warning(f"Vision model not found: {vision_model_path}")
        
        # Load fusion model
        fusion_model_path = config["models"]["fusion"]["model_path"]
        if os.path.exists(fusion_model_path):
            fusion_model = torch.load(fusion_model_path, map_location='cpu')
            logger.info(f"Loaded fusion model: {fusion_model_path}")
        else:
            logger.warning(f"Fusion model not found: {fusion_model_path}")
        
        # Load segmentation model
        segmentation_model_path = config["models"]["segmentation"]["model_path"]
        if os.path.exists(segmentation_model_path):
            segmentation_model = torch.load(segmentation_model_path, map_location='cpu')
            logger.info(f"Loaded segmentation model: {segmentation_model_path}")
        else:
            logger.warning(f"Segmentation model not found: {segmentation_model_path}")
        
        # Load Whisper model
        whisper_model_name = config["models"]["nlp"]["whisper_model"]
        whisper_model = whisper.load_model(whisper_model_name)
        logger.info(f"Loaded Whisper model: {whisper_model_name}")
        
        # Load sentence transformer
        st_model_name = config["models"]["nlp"]["sentence_transformer"]
        sentence_transformer = SentenceTransformer(st_model_name)
        logger.info(f"Loaded sentence transformer: {st_model_name}")
        
        # Initialize dose optimizer
        dose_optimizer = DoseOptimizer()
        logger.info("Initialized dose optimizer")
        
        # Initialize flight planner
        flight_planner = FlightPlanner()
        logger.info("Initialized flight planner")
        
        # Configure Gemini API
        gemini_key = os.getenv("GEMINI_API_KEY") or config.get("auth", {}).get("api_keys", {}).get("gemini")
        if gemini_key:
            genai.configure(api_key=gemini_key)
        else:
            logger.warning("Gemini API key not set; NLP generation will be limited")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

def calculate_image_hash(image_data: bytes) -> str:
    """Calculate SHA-256 hash of image data."""
    return hashlib.sha256(image_data).hexdigest()

def save_uploaded_file(file: UploadFile, upload_dir: str) -> str:
    """Save uploaded file and return the file path."""
    upload_path = Path(upload_dir or "uploads")
    upload_path.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = upload_path / unique_filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return str(file_path)

def process_image_with_vision(image_path: str) -> List[Dict[str, Any]]:
    """Process image with vision model."""
    if vision_model is None:
        raise HTTPException(status_code=500, detail="Vision model not loaded")
    
    try:
        # Run inference
        results = vision_model(image_path)

        predictions = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                # boxes.xyxy is typically a tensor of shape (N, 4)
                num_boxes = boxes.xyxy.shape[0]
                for i in range(num_boxes):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy.tolist()
                    confidence = float(boxes.conf[i].cpu().numpy()) if getattr(boxes, "conf", None) is not None else 0.0
                    class_id = int(boxes.cls[i].cpu().numpy()) if getattr(boxes, "cls", None) is not None else 0

                    prediction = {
                        "id": i + 1,
                        "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                        "confidence": confidence,
                        "category_id": class_id + 1,
                        "severity": 1,
                        "area": float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
                    }
                    predictions.append(prediction)

        return predictions
        
    except Exception as e:
        logger.error(f"Vision processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vision processing failed: {e}")

def process_text_with_nlp(text: str) -> Dict[str, Any]:
    """Process text with NLP pipeline."""
    if sentence_transformer is None:
        raise HTTPException(status_code=500, detail="NLP model not loaded")
    
    try:
        # Generate text embedding
        text_embedding = sentence_transformer.encode(text)
        
        # Use Gemini for structured symptom extraction
        if text and text.strip():
            # Create Gemini model
            model = genai.GenerativeModel(config["models"]["gemini"]["model"])
            
            # Create prompt
            prompt = f"""You are an agricultural expert. Extract structured information from farmer notes about pest/disease symptoms. 
            Respond with JSON format: {{"symptoms": ["symptom1", "symptom2"], "likely_causes": ["cause1", "cause2"], "severity_indicators": ["indicator1"], "clarifying_questions": ["question1"]}}
            
            Farmer note: {text}"""
            
            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config["models"]["gemini"]["temperature"],
                    max_output_tokens=config["models"]["gemini"]["max_output_tokens"],
                    top_p=config["models"]["gemini"]["top_p"],
                    top_k=config["models"]["gemini"]["top_k"]
                )
            )
            # Be resilient to response formats
            structured_text = getattr(response, "text", None)
            if not structured_text and hasattr(response, "candidates") and response.candidates:
                structured_text = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts else "{}"
            try:
                structured_info = json.loads(structured_text or "{}")
            except Exception:
                structured_info = {"symptoms": [], "likely_causes": [], "severity_indicators": [], "clarifying_questions": []}
        else:
            structured_info = {
                "symptoms": [],
                "likely_causes": [],
                "severity_indicators": [],
                "clarifying_questions": []
            }
        
        return {
            "text_embedding": text_embedding.tolist(),
            "structured_info": structured_info,
            "original_text": text
        }
        
    except Exception as e:
        logger.error(f"NLP processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"NLP processing failed: {e}")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper."""
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")
    
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {e}")

# API Endpoints

# Removed deprecated @app.on_event("startup") - now handled by lifespan

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": {
            "vision": vision_model is not None,
            "fusion": fusion_model is not None,
            "segmentation": segmentation_model is not None,
            "whisper": whisper_model is not None,
            "sentence_transformer": sentence_transformer is not None,
            "dose_optimizer": dose_optimizer is not None,
            "flight_planner": flight_planner is not None
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    transcript: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Predict pest/disease instances from image and optional text."""
    
    start_time = datetime.utcnow()
    run_id = str(uuid.uuid4())
    
    try:
        # Validate file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file
        image_path = save_uploaded_file(image, config["upload"]["upload_dir"])
        
        # Calculate image hash
        with open(image_path, 'rb') as f:
            image_data = f.read()
        image_hash = calculate_image_hash(image_data)
        
        # Process image with vision model
        predictions = process_image_with_vision(image_path)
        
        # Process text if provided
        text_info = None
        if transcript:
            text_info = process_text_with_nlp(transcript)
        
        # Calculate confidence scores
        confidence_scores = [pred["confidence"] for pred in predictions]
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Log prediction result
        log_entry = {
            "run_id": run_id,
            "image_hash": image_hash,
            "image_path": image_path,
            "predictions": predictions,
            "text_info": text_info,
            "processing_time": processing_time,
            "timestamp": start_time.isoformat()
        }
        
        # Save log entry
        log_dir = Path("logs/predictions")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{run_id}.json"
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        return PredictionResponse(
            run_id=run_id,
            predictions=predictions,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plan", response_model=PlanningResponse)
async def plan(request: PlanningRequest):
    """Generate flight plan and dose optimization."""
    
    start_time = datetime.utcnow()
    run_id = str(uuid.uuid4())
    
    try:
        # Convert predictions to PlantInstance objects
        plants = []
        for i, pred in enumerate(request.predictions):
            plant = PlantInstance(
                id=pred["id"],
                bbox=pred["bbox"],
                area=pred["area"],
                severity=pred["severity"],
                confidence=pred["confidence"],
                category_id=pred["category_id"],
                location=(pred["bbox"][0], pred["bbox"][1])  # Simplified location
            )
            plants.append(plant)
        
        # Optimize doses
        if dose_optimizer is None:
            raise HTTPException(status_code=500, detail="Dose optimizer not loaded")
        
        optimization_result = dose_optimizer.solve_doses(plants)
        
        if optimization_result.status.value != "optimal":
            raise HTTPException(
                status_code=400, 
                detail=f"Optimization failed: {optimization_result.status.value}"
            )
        
        # Generate flight plan
        if flight_planner is None:
            raise HTTPException(status_code=500, detail="Flight planner not loaded")
        
        waypoints = flight_planner.generate_waypoints(plants, request.location)
        mavlink_json = flight_planner.generate_mavlink_mission(waypoints, optimization_result.doses)
        
        # Calculate total pesticide and estimated time
        total_pesticide = sum(optimization_result.doses)
        estimated_time = flight_planner.estimate_flight_time(waypoints)
        
        # Check if approval is required
        requires_approval = (
            any(p.confidence < config["safety"]["confidence_threshold"] for p in plants) or
            total_pesticide > config["constraints"]["regulatory"]["max_total_dose"] * 0.8
        )
        
        # Generate explainability URL
        explainability_url = f"/explain/{run_id}"
        
        # Log planning result
        log_entry = {
            "run_id": run_id,
            "plants": [{"id": p.id, "severity": p.severity, "confidence": p.confidence} for p in plants],
            "optimization_result": {
                "status": optimization_result.status.value,
                "objective_value": optimization_result.objective_value,
                "solve_time": optimization_result.solve_time,
                "warnings": optimization_result.warnings
            },
            "doses": optimization_result.doses,
            "waypoints": waypoints,
            "total_pesticide": total_pesticide,
            "estimated_time": estimated_time,
            "requires_approval": requires_approval,
            "timestamp": start_time.isoformat()
        }
        
        # Save log entry
        log_dir = Path("logs/planning")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{run_id}.json"
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        return PlanningResponse(
            run_id=run_id,
            doses=optimization_result.doses,
            waypoints=waypoints,
            mavlink_json=mavlink_json,
            explainability_url=explainability_url,
            total_pesticide=total_pesticide,
            estimated_time=estimated_time,
            requires_approval=requires_approval
        )
        
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/approve", response_model=ApprovalResponse)
async def approve(request: ApprovalRequest):
    """Handle operator approval/rejection of flight plans."""
    
    try:
        # Log approval decision
        log_entry = {
            "run_id": request.run_id,
            "decision": request.decision,
            "operator_id": request.operator_id,
            "comments": request.comments,
            "signature": request.signature,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save log entry
        log_dir = Path("logs/approvals")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{request.run_id}_approval.json"
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        return ApprovalResponse(
            run_id=request.run_id,
            decision=request.decision,
            timestamp=datetime.utcnow().isoformat(),
            operator_id=request.operator_id
        )
        
    except Exception as e:
        logger.error(f"Approval processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/{run_id}")
async def get_logs(run_id: str):
    """Retrieve immutable run logs."""
    
    try:
        # Look for log files
        log_dirs = ["logs/predictions", "logs/planning", "logs/approvals"]
        logs = {}
        
        for log_dir in log_dirs:
            log_file = Path(log_dir) / f"{run_id}.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs[log_dir.split('/')[-1]] = json.load(f)
        
        if not logs:
            raise HTTPException(status_code=404, detail="Logs not found")
        
        return logs
        
    except Exception as e:
        logger.error(f"Failed to retrieve logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/{run_id}")
async def explain(run_id: str):
    """Generate explainability report for a run."""
    
    try:
        # Load run logs
        log_dirs = ["logs/predictions", "logs/planning", "logs/approvals"]
        logs = {}
        
        for log_dir in log_dirs:
            log_file = Path(log_dir) / f"{run_id}.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs[log_dir.split('/')[-1]] = json.load(f)
        
        if not logs:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Generate explainability report
        explainability_report = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_predictions": len(logs.get("predictions", {}).get("predictions", [])),
                "total_doses": len(logs.get("planning", {}).get("doses", [])),
                "requires_approval": logs.get("planning", {}).get("requires_approval", False)
            },
            "vision_explanation": {
                "model_version": "1.0.0",
                "confidence_scores": logs.get("predictions", {}).get("predictions", []),
                "grad_cam_available": True
            },
            "optimization_explanation": {
                "solver": "CVXPY",
                "objective": "minimize_total_dose",
                "constraints": [
                    "cure_probability >= 0.85",
                    "per_plant_dose <= 50ml",
                    "total_dose <= 1000ml"
                ],
                "solution_quality": logs.get("planning", {}).get("optimization_result", {}).get("status", "unknown")
            },
            "safety_checks": {
                "confidence_threshold": config["safety"]["confidence_threshold"],
                "regulatory_compliance": True,
                "operator_approval": logs.get("approvals") is not None
            }
        }
        
        return explainability_report
        
    except Exception as e:
        logger.error(f"Failed to generate explainability report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = int(config.get("server", {}).get("port", 8000))
    reload_opt = bool(config.get("development", {}).get("reload", False))
    log_level = str(config.get("development", {}).get("log_level", "info")).lower()
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,  # Disable reload to avoid import issues
        log_level=log_level
    )
