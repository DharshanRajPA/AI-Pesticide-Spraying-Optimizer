from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from ultralytics import YOLO
import json
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLO model
model_path = "models/best.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
else:
    model = None
    print("WARNING: Model not found, using mock predictions")

@app.get("/")
async def read_root():
    return HTMLResponse(open("static/index.html").read())

@app.post("/detect")
async def detect_pests(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if model:
            # Real YOLO detection
            results = model(image)
            detections = results[0].boxes
            pest_count = len(detections) if detections is not None else 0
            pest_types = ["pest"] * pest_count  # Simplified
        else:
            # Mock detection for demo
            pest_count = 3
            pest_types = ["aphid", "caterpillar", "beetle"]
        
        # Calculate spraying recommendations
        field_area = estimate_field_area(image)
        pesticide_quantity = calculate_pesticide_quantity(field_area, pest_count)
        cost_estimate = calculate_cost(pesticide_quantity)
        
        return {
            "pest_count": pest_count,
            "pest_types": pest_types,
            "field_area": field_area,
            "pesticide_quantity": pesticide_quantity,
            "cost_estimate": cost_estimate,
            "recommendations": generate_recommendations(pest_count, field_area)
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

def estimate_field_area(image):
    # Simple area estimation based on image dimensions
    height, width = image.shape[:2]
    return round((height * width) / 10000, 2)  # Rough estimate in hectares

def calculate_pesticide_quantity(field_area, pest_count):
    # Simple calculation: 2ml per hectare per pest
    return round(field_area * pest_count * 2, 2)

def calculate_cost(quantity):
    # Simple cost: $5 per ml
    return round(quantity * 5, 2)

def generate_recommendations(pest_count, field_area):
    if pest_count == 0:
        return "No pests detected. No spraying needed."
    elif pest_count < 5:
        return "Light infestation. Consider organic pesticides."
    else:
        return "Heavy infestation. Use chemical pesticides immediately."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)