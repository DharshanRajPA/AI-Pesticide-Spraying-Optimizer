#!/usr/bin/env python3
"""
AgriSprayAI - Simple Pest Detection & Spraying Optimization
A clean, minimal prototype for mentor demonstration.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ML and Computer Vision
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AgriSprayAI",
    description="Simple Pest Detection & Spraying Optimization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
vision_model = None

# Pydantic models
class PredictionResponse:
    """Response model for prediction endpoint."""
    def __init__(self, run_id: str, predictions: List[Dict], confidence_scores: List[float], 
                 processing_time: float, pesticide_recommendation: Dict, cost_estimate: Dict):
        self.run_id = run_id
        self.predictions = predictions
        self.confidence_scores = confidence_scores
        self.processing_time = processing_time
        self.pesticide_recommendation = pesticide_recommendation
        self.cost_estimate = cost_estimate

def load_models():
    """Load the YOLO model."""
    global vision_model
    
    try:
        # Try to load the best model first, fallback to yolov8n.pt
        model_paths = ["models/segmentation_best.pt", "yolov8n.pt"]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                vision_model = YOLO(model_path)
                logger.info(f"Loaded vision model: {model_path}")
                return
        
        # If no model found, create a mock model for demonstration
        logger.warning("No model found, using mock model for demonstration")
        vision_model = "mock"
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        vision_model = "mock"

def calculate_spraying_recommendations(predictions: List[Dict]) -> Dict:
    """Calculate pesticide recommendations based on predictions."""
    if not predictions:
        return {
            "pesticide_type": "No pests detected",
            "quantity_ml": 0,
            "coverage_area_sqm": 0,
            "application_method": "None required"
        }
    
    # Simple calculation based on pest count and confidence
    total_pests = len(predictions)
    avg_confidence = sum(p["confidence"] for p in predictions) / total_pests if predictions else 0
    
    # Determine pesticide type based on pest count
    if total_pests <= 2:
        pesticide_type = "Organic Neem Oil"
        base_quantity = 50
    elif total_pests <= 5:
        pesticide_type = "Pyrethrin-based Spray"
        base_quantity = 100
    else:
        pesticide_type = "Synthetic Pesticide"
        base_quantity = 150
    
    # Adjust quantity based on confidence
    quantity_ml = int(base_quantity * (1 + avg_confidence))
    coverage_area_sqm = total_pests * 2  # 2 sqm per pest
    
    return {
        "pesticide_type": pesticide_type,
        "quantity_ml": quantity_ml,
        "coverage_area_sqm": coverage_area_sqm,
        "application_method": "Spray evenly over affected areas"
    }

def calculate_cost_estimate(recommendation: Dict) -> Dict:
    """Calculate cost estimate for pesticide application."""
    quantity_ml = recommendation["quantity_ml"]
    
    # Cost per ml (rough estimates)
    cost_per_ml = {
        "Organic Neem Oil": 0.05,
        "Pyrethrin-based Spray": 0.08,
        "Synthetic Pesticide": 0.12
    }
    
    pesticide_type = recommendation["pesticide_type"]
    if pesticide_type in cost_per_ml:
        total_cost = quantity_ml * cost_per_ml[pesticide_type]
    else:
        total_cost = 0
    
    return {
        "pesticide_cost": round(total_cost, 2),
        "labor_cost": 25.0,  # Fixed labor cost
        "total_cost": round(total_cost + 25.0, 2),
        "cost_per_sqm": round((total_cost + 25.0) / max(recommendation["coverage_area_sqm"], 1), 2)
    }

def process_image_with_vision(image_path: str) -> List[Dict[str, Any]]:
    """Process image with vision model."""
    if vision_model == "mock":
        # Return mock predictions for demonstration
        return [
            {
                "id": 1,
                "bbox": [100, 100, 50, 50],
                "confidence": 0.85,
                "category_id": 1,
                "category_name": "Aphid",
                "severity": 2,
                "area": 2500
            },
            {
                "id": 2,
                "bbox": [200, 150, 40, 40],
                "confidence": 0.72,
                "category_id": 2,
                "category_name": "Caterpillar",
                "severity": 3,
                "area": 1600
            }
        ]
    
    if vision_model is None:
        raise HTTPException(status_code=500, detail="Vision model not loaded")
    
    try:
        # Run inference
        results = vision_model(image_path)
        
        predictions = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                num_boxes = boxes.xyxy.shape[0]
                for i in range(num_boxes):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy.tolist()
                    confidence = float(boxes.conf[i].cpu().numpy()) if getattr(boxes, "conf", None) is not None else 0.0
                    class_id = int(boxes.cls[i].cpu().numpy()) if getattr(boxes, "cls", None) is not None else 0
                    
                    # Map class_id to pest names
                    pest_names = ["Aphid", "Caterpillar", "Beetle", "Moth", "Wasp", "Grasshopper"]
                    category_name = pest_names[class_id] if class_id < len(pest_names) else f"Pest_{class_id}"
                    
                    prediction = {
                        "id": i + 1,
                        "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                        "confidence": confidence,
                        "category_id": class_id + 1,
                        "category_name": category_name,
                        "severity": min(5, max(1, int(confidence * 5))),
                        "area": float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
                    }
                    predictions.append(prediction)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Vision processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vision processing failed: {e}")

def save_uploaded_file(file: UploadFile, upload_dir: str = "uploads") -> str:
    """Save uploaded file and return the file path."""
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = upload_path / unique_filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return str(file_path)

# Load models on startup
load_models()

# API Endpoints

@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AgriSprayAI</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c5530; text-align: center; }
                .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
                .upload-area:hover { border-color: #2c5530; background: #f9f9f9; }
                button { background: #2c5530; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
                button:hover { background: #1e3a21; }
                .results { margin-top: 20px; padding: 20px; background: #f0f8f0; border-radius: 10px; display: none; }
                .pest-item { background: white; margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 4px solid #2c5530; }
                .recommendation { background: #e8f4e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .cost { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌾 AgriSprayAI - Pest Detection & Spraying Optimization</h1>
                
                <div class="upload-area">
                    <h3>Upload Field Image</h3>
                    <input type="file" id="imageInput" accept="image/*" style="margin: 10px;">
                    <br>
                    <button onclick="detectPests()">🔍 Detect Pests & Get Recommendations</button>
                </div>
                
                <div id="results" class="results">
                    <h3>Detection Results & Recommendations</h3>
                    <div id="pestList"></div>
                    <div id="recommendations"></div>
                    <div id="costEstimate"></div>
                </div>
            </div>

            <script>
                async function detectPests() {
                    const file = document.getElementById('imageInput').files[0];
                    if (!file) {
                        alert('Please select an image file');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        const response = await fetch('/detect', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error('Detection failed');
                        }
                        
                        const results = await response.json();
                        displayResults(results);
                    } catch (error) {
                        alert('Error: ' + error.message);
                    }
                }
                
                function displayResults(data) {
                    document.getElementById('results').style.display = 'block';
                    
                    // Display pest detections
                    const pestList = document.getElementById('pestList');
                    if (data.predictions && data.predictions.length > 0) {
                        pestList.innerHTML = '<h4>Detected Pests:</h4>';
                        data.predictions.forEach(pest => {
                            pestList.innerHTML += `
                                <div class="pest-item">
                                    <strong>${pest.category_name}</strong> (Confidence: ${(pest.confidence * 100).toFixed(1)}%)
                                    <br>Severity: ${pest.severity}/5 | Area: ${pest.area.toFixed(0)} pixels
                                </div>
                            `;
                        });
                    } else {
                        pestList.innerHTML = '<h4>No pests detected! 🎉</h4>';
                    }
                    
                    // Display recommendations
                    const recommendations = document.getElementById('recommendations');
                    recommendations.innerHTML = `
                        <div class="recommendation">
                            <h4>Spraying Recommendation:</h4>
                            <p><strong>Pesticide:</strong> ${data.pesticide_recommendation.pesticide_type}</p>
                            <p><strong>Quantity:</strong> ${data.pesticide_recommendation.quantity_ml}ml</p>
                            <p><strong>Coverage:</strong> ${data.pesticide_recommendation.coverage_area_sqm} sqm</p>
                            <p><strong>Method:</strong> ${data.pesticide_recommendation.application_method}</p>
                        </div>
                    `;
                    
                    // Display cost estimate
                    const costEstimate = document.getElementById('costEstimate');
                    costEstimate.innerHTML = `
                        <div class="cost">
                            <h4>Cost Estimate:</h4>
                            <p><strong>Pesticide Cost:</strong> $${data.cost_estimate.pesticide_cost}</p>
                            <p><strong>Labor Cost:</strong> $${data.cost_estimate.labor_cost}</p>
                            <p><strong>Total Cost:</strong> $${data.cost_estimate.total_cost}</p>
                            <p><strong>Cost per sqm:</strong> $${data.cost_estimate.cost_per_sqm}</p>
                        </div>
                    `;
                }
            </script>
        </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": vision_model is not None,
        "version": "1.0.0"
    }

@app.post("/detect")
async def detect_pests(file: UploadFile = File(...)):
    """Detect pests in uploaded image and provide spraying recommendations."""
    
    start_time = datetime.utcnow()
    run_id = str(uuid.uuid4())
    
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_extension = Path(file.filename).suffix
        unique_filename = f"{run_id}{file_extension}"
        image_path = os.path.join(upload_dir, unique_filename)
        
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image with vision model
        predictions = process_image_with_vision(image_path)
        
        # Calculate spraying recommendations
        pesticide_recommendation = calculate_spraying_recommendations(predictions)
        
        # Calculate cost estimate
        cost_estimate = calculate_cost_estimate(pesticide_recommendation)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Clean up uploaded file
        try:
            os.remove(image_path)
        except:
            pass
        
        return {
            "run_id": run_id,
            "predictions": predictions,
            "confidence_scores": [p["confidence"] for p in predictions],
            "processing_time": processing_time,
            "pesticide_recommendation": pesticide_recommendation,
            "cost_estimate": cost_estimate
        }
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🌾 Starting AgriSprayAI...")
    print("📱 Open: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)