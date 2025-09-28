from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
import uuid
from datetime import datetime

app = FastAPI(title="AgriSprayAI", description="AI-Powered Pest Detection with User Query Processing")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/analyze")
async def analyze_field(
    file: UploadFile = File(...),
    user_query: str = Form(...)
):
    """
    Analyze field image AND user query together to provide comprehensive pest detection and recommendations.
    """
    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image with YOLO
        image_analysis = process_image_analysis(image)
        
        # Process user query
        query_analysis = process_user_query(user_query)
        
        # Combine image and query analysis
        combined_analysis = combine_analyses(image_analysis, query_analysis)
        
        # Generate comprehensive recommendations
        recommendations = generate_comprehensive_recommendations(combined_analysis)
        
        # Calculate costs and spraying plan
        spraying_plan = calculate_spraying_plan(combined_analysis)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "analysis_id": analysis_id,
            "timestamp": start_time.isoformat(),
            "processing_time": processing_time,
            "user_query": user_query,
            "image_analysis": image_analysis,
            "query_analysis": query_analysis,
            "combined_analysis": combined_analysis,
            "recommendations": recommendations,
            "spraying_plan": spraying_plan,
            "confidence_score": calculate_confidence_score(combined_analysis)
        }
        
    except Exception as e:
        return {"error": str(e), "analysis_id": str(uuid.uuid4())}

@app.post("/detect")
async def detect_pests(file: UploadFile = File(...)):
    """Legacy endpoint for image-only detection"""
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

def process_image_analysis(image):
    """Process image using YOLO model"""
    if model:
        # Real YOLO detection
        results = model(image)
        detections = results[0].boxes
        
        if detections is not None:
            pest_count = len(detections)
            pest_types = []
            confidences = []
            
            for i in range(pest_count):
                # Get class and confidence
                class_id = int(detections.cls[i].item())
                confidence = float(detections.conf[i].item())
                
                # Map class to pest name
                pest_names = ["Aphid", "Caterpillar", "Beetle", "Moth", "Wasp", "Grasshopper", "Snail", "Slug"]
                pest_name = pest_names[class_id] if class_id < len(pest_names) else f"Pest_{class_id}"
                
                pest_types.append(pest_name)
                confidences.append(confidence)
        else:
            pest_count = 0
            pest_types = []
            confidences = []
    else:
        # Mock detection for demo
        pest_count = 3
        pest_types = ["Aphid", "Caterpillar", "Beetle"]
        confidences = [0.85, 0.72, 0.68]
    
    # Estimate field area
    field_area = estimate_field_area(image)
    
    return {
        "pest_count": pest_count,
        "pest_types": pest_types,
        "confidences": confidences,
        "field_area": field_area,
        "image_quality": "good" if image.shape[0] > 500 else "poor"
    }

def process_user_query(query):
    """Process user text query to extract pest information and symptoms"""
    query_lower = query.lower()
    
    # Extract pest mentions
    pest_keywords = {
        "aphid": ["aphid", "aphids", "greenfly", "blackfly"],
        "caterpillar": ["caterpillar", "caterpillars", "worm", "worms", "larva"],
        "beetle": ["beetle", "beetles", "bug", "bugs"],
        "moth": ["moth", "moths"],
        "wasp": ["wasp", "wasps"],
        "grasshopper": ["grasshopper", "grasshoppers", "locust"],
        "snail": ["snail", "snails"],
        "slug": ["slug", "slugs"]
    }
    
    detected_pests = []
    for pest, keywords in pest_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_pests.append(pest)
    
    # Extract symptoms
    symptom_keywords = {
        "holes_in_leaves": ["holes", "chewed", "eaten", "damaged leaves"],
        "yellowing": ["yellow", "yellowing", "discolored"],
        "wilting": ["wilt", "wilting", "drooping"],
        "spots": ["spots", "patches", "marks"],
        "stunted_growth": ["stunted", "small", "not growing"]
    }
    
    detected_symptoms = []
    for symptom, keywords in symptom_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_symptoms.append(symptom)
    
    # Determine severity from query
    severity_indicators = {
        "low": ["few", "some", "little", "slight"],
        "medium": ["many", "several", "moderate"],
        "high": ["lots", "many", "severe", "heavy", "infestation"]
    }
    
    severity = "medium"  # default
    for level, indicators in severity_indicators.items():
        if any(indicator in query_lower for indicator in indicators):
            severity = level
            break
    
    return {
        "detected_pests": detected_pests,
        "detected_symptoms": detected_symptoms,
        "severity": severity,
        "query_length": len(query),
        "urgency": "high" if any(word in query_lower for word in ["urgent", "emergency", "help", "quickly"]) else "normal"
    }

def combine_analyses(image_analysis, query_analysis):
    """Combine image and query analysis for comprehensive results"""
    
    # Merge pest detections
    image_pests = set(image_analysis["pest_types"])
    query_pests = set(query_analysis["detected_pests"])
    
    # Combine pest lists (prioritize image detection)
    all_pests = list(image_pests.union(query_pests))
    confirmed_pests = list(image_pests.intersection(query_pests))
    
    # Calculate combined confidence
    if image_analysis["pest_count"] > 0 and query_analysis["detected_pests"]:
        confidence = 0.9  # High confidence when both agree
    elif image_analysis["pest_count"] > 0 or query_analysis["detected_pests"]:
        confidence = 0.7  # Medium confidence when one source detects
    else:
        confidence = 0.3  # Low confidence when neither detects
    
    # Determine overall severity
    if query_analysis["severity"] == "high" or image_analysis["pest_count"] > 5:
        overall_severity = "high"
    elif query_analysis["severity"] == "low" and image_analysis["pest_count"] < 3:
        overall_severity = "low"
    else:
        overall_severity = "medium"
    
    return {
        "all_detected_pests": all_pests,
        "confirmed_pests": confirmed_pests,
        "image_pests": list(image_pests),
        "query_pests": list(query_pests),
        "total_pest_count": max(image_analysis["pest_count"], len(query_analysis["detected_pests"])),
        "symptoms": query_analysis["detected_symptoms"],
        "severity": overall_severity,
        "confidence": confidence,
        "field_area": image_analysis["field_area"],
        "urgency": query_analysis["urgency"]
    }

def generate_comprehensive_recommendations(combined_analysis):
    """Generate comprehensive recommendations based on combined analysis"""
    
    pest_count = combined_analysis["total_pest_count"]
    severity = combined_analysis["severity"]
    symptoms = combined_analysis["symptoms"]
    urgency = combined_analysis["urgency"]
    
    recommendations = []
    
    # Immediate actions
    if urgency == "high":
        recommendations.append("🚨 URGENT: Immediate action required!")
    
    # Pest-specific recommendations
    if "aphid" in combined_analysis["all_detected_pests"]:
        recommendations.append("🐛 Aphids detected: Use neem oil or insecticidal soap")
    
    if "caterpillar" in combined_analysis["all_detected_pests"]:
        recommendations.append("🐛 Caterpillars detected: Apply Bt (Bacillus thuringiensis)")
    
    if "beetle" in combined_analysis["all_detected_pests"]:
        recommendations.append("🐛 Beetles detected: Use pyrethrin-based spray")
    
    # Symptom-based recommendations
    if "holes_in_leaves" in symptoms:
        recommendations.append("🍃 Leaf damage detected: Apply contact insecticide")
    
    if "yellowing" in symptoms:
        recommendations.append("🟡 Yellowing detected: Check for nutrient deficiency or disease")
    
    # Severity-based recommendations
    if severity == "high":
        recommendations.append("⚠️ High infestation: Use chemical pesticides immediately")
        recommendations.append("📅 Schedule follow-up treatment in 7-10 days")
    elif severity == "medium":
        recommendations.append("⚖️ Moderate infestation: Consider organic options first")
    else:
        recommendations.append("✅ Light infestation: Monitor and use preventive measures")
    
    # General recommendations
    recommendations.append("🌱 Apply treatment early morning or late evening")
    recommendations.append("💧 Ensure proper coverage of all plant surfaces")
    
    return recommendations

def calculate_spraying_plan(combined_analysis):
    """Calculate detailed spraying plan"""
    
    pest_count = combined_analysis["total_pest_count"]
    field_area = combined_analysis["field_area"]
    severity = combined_analysis["severity"]
    
    # Calculate pesticide quantity based on severity and area
    if severity == "high":
        base_rate = 3.0  # ml per hectare
    elif severity == "medium":
        base_rate = 2.0
    else:
        base_rate = 1.0
    
    pesticide_quantity = round(field_area * base_rate * pest_count, 2)
    
    # Select pesticide type
    if severity == "high":
        pesticide_type = "Chemical Pesticide (Pyrethroid)"
        cost_per_ml = 0.15
    elif severity == "medium":
        pesticide_type = "Organic Pesticide (Neem Oil)"
        cost_per_ml = 0.08
    else:
        pesticide_type = "Preventive Treatment (Soap Solution)"
        cost_per_ml = 0.03
    
    # Calculate costs
    pesticide_cost = round(pesticide_quantity * cost_per_ml, 2)
    labor_cost = 25.0
    total_cost = round(pesticide_cost + labor_cost, 2)
    
    return {
        "pesticide_type": pesticide_type,
        "quantity_ml": pesticide_quantity,
        "field_area_hectares": field_area,
        "application_rate": f"{base_rate}ml per hectare",
        "pesticide_cost": pesticide_cost,
        "labor_cost": labor_cost,
        "total_cost": total_cost,
        "estimated_time_hours": round(field_area * 0.5, 1),
        "best_application_time": "Early morning (6-8 AM) or late evening (6-8 PM)"
    }

def calculate_confidence_score(combined_analysis):
    """Calculate overall confidence score"""
    
    base_confidence = combined_analysis["confidence"]
    
    # Boost confidence if both image and query agree
    if combined_analysis["confirmed_pests"]:
        base_confidence += 0.1
    
    # Boost confidence if symptoms match pest types
    if combined_analysis["symptoms"] and combined_analysis["all_detected_pests"]:
        base_confidence += 0.05
    
    return min(1.0, round(base_confidence, 2))

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