# 🌾 AgriSprayAI - Pest Detection & Spraying Optimization

A simple, clean prototype for AI-powered pest detection and pesticide spraying optimization.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python start.py
```

### 3. Open in Browser
Go to: http://localhost:8000

### 4. Upload Image & Get Results
- Upload a field image
- Get pest detection results
- Receive spraying recommendations
- View cost estimates

## 🎯 Features

- **Pest Detection**: AI-powered identification of pests in field images
- **Spraying Optimization**: Smart recommendations for pesticide application
- **Cost Estimation**: Calculate costs for pesticide and labor
- **Simple Interface**: Clean, easy-to-use web interface

## 📁 Project Structure

```
AgriSprayAI/
├── app.py              # Main FastAPI application
├── start.py            # Simple startup script
├── requirements.txt    # Minimal dependencies
├── README.md          # This file
├── models/            # ML models (if available)
└── uploads/           # Temporary upload directory
```

## 🔧 Technical Details

- **Backend**: FastAPI with YOLO for pest detection
- **Frontend**: Simple HTML/JavaScript interface
- **ML Model**: YOLOv8 for object detection
- **Optimization**: Smart pesticide quantity calculation

## 🎯 Success Criteria

✅ **Single Command Startup**: `python start.py`  
✅ **Web Interface**: http://localhost:8000  
✅ **Image Upload**: Upload field images  
✅ **Pest Detection**: AI identifies pests  
✅ **Recommendations**: Get spraying advice  
✅ **Cost Estimates**: Calculate application costs  

## 🚨 Troubleshooting

- **Model Not Found**: The app will use mock data for demonstration
- **Port Already in Use**: Change port in `start.py` if needed
- **Missing Dependencies**: Run `pip install -r requirements.txt`

## 📝 Notes

This is a **minimal prototype** designed for demonstration purposes. It includes:
- Core pest detection functionality
- Simple spraying recommendations
- Cost estimation
- Clean, understandable code

Perfect for mentor demonstrations and proof-of-concept validation.