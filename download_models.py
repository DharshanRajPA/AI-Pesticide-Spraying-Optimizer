#!/usr/bin/env python3
"""
Model Download Script for AgriSprayAI
Downloads and sets up the required AI models for the system.
"""

import os
import sys
import requests
import zipfile
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for models."""
    logger.info("📁 Creating model directories...")
    
    directories = [
        "models",
        "models/yolov8_baseline",
        "models/yolov8_baseline/weights",
        "models/fusion",
        "models/segmentation"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def download_yolov8_model():
    """Download YOLOv8 baseline model."""
    logger.info("🤖 Downloading YOLOv8 baseline model...")
    
    try:
        from ultralytics import YOLO
        
        # Download YOLOv8n (nano) model - smallest and fastest
        model = YOLO('yolov8n.pt')
        
        # Save to our models directory
        model_path = "models/yolov8_baseline/weights/best.pt"
        shutil.copy2('yolov8n.pt', model_path)
        
        logger.info(f"✅ YOLOv8 model downloaded and saved to: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download YOLOv8 model: {e}")
        return False

def create_fusion_model_placeholder():
    """Create a placeholder fusion model."""
    logger.info("🔗 Creating fusion model placeholder...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple fusion model architecture
        class SimpleFusionModel(nn.Module):
            def __init__(self, vision_dim=1024, text_dim=384, output_dim=256):
                super().__init__()
                self.vision_proj = nn.Linear(vision_dim, output_dim)
                self.text_proj = nn.Linear(text_dim, output_dim)
                self.fusion = nn.Sequential(
                    nn.Linear(output_dim * 2, output_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(output_dim, 12)  # 12 classes for agricultural pests
                )
                
            def forward(self, vision_features, text_features):
                v_proj = self.vision_proj(vision_features)
                t_proj = self.text_proj(text_features)
                fused = torch.cat([v_proj, t_proj], dim=-1)
                return self.fusion(fused)
        
        # Create and save the model
        model = SimpleFusionModel()
        model_path = "models/fusion_model.pt"
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"✅ Fusion model placeholder created: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create fusion model: {e}")
        return False

def create_segmentation_model_placeholder():
    """Create a placeholder segmentation model."""
    logger.info("🎯 Creating segmentation model placeholder...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple segmentation model
        class SimpleSegmentationModel(nn.Module):
            def __init__(self, num_classes=12):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU()
                )
                self.classifier = nn.Conv2d(256, num_classes, 1)
                
            def forward(self, x):
                features = self.backbone(x)
                return self.classifier(features)
        
        # Create and save the model
        model = SimpleSegmentationModel()
        model_path = "models/segmentation_best.pt"
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"✅ Segmentation model placeholder created: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create segmentation model: {e}")
        return False

def download_pretrained_models():
    """Download additional pretrained models if needed."""
    logger.info("📥 Downloading additional pretrained models...")
    
    try:
        # Download sentence transformer model (this will be cached automatically)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("✅ Sentence transformer model ready")
        
        # Download Whisper model (this will be cached automatically)
        import whisper
        model = whisper.load_model("base")
        logger.info("✅ Whisper model ready")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download pretrained models: {e}")
        return False

def create_model_config():
    """Create model configuration file."""
    logger.info("⚙️ Creating model configuration...")
    
    config = {
        "models": {
            "yolov8": {
                "path": "models/yolov8_baseline/weights/best.pt",
                "type": "detection",
                "classes": [
                    "ants", "bees", "beetle", "caterpillar", "earthworms", 
                    "earwig", "grasshopper", "moth", "slug", "snail", "wasp", "weevil"
                ],
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45
            },
            "fusion": {
                "path": "models/fusion_model.pt",
                "type": "multimodal_fusion",
                "input_dimensions": {
                    "vision": 1024,
                    "text": 384
                },
                "output_classes": 12
            },
            "segmentation": {
                "path": "models/segmentation_best.pt",
                "type": "segmentation",
                "classes": 12,
                "input_size": [640, 640]
            }
        },
        "download_info": {
            "yolov8": "Downloaded from Ultralytics YOLOv8",
            "fusion": "Custom fusion model for multimodal processing",
            "segmentation": "Custom segmentation model for instance masks"
        }
    }
    
    import json
    with open("models/model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("✅ Model configuration created: models/model_config.json")

def verify_models():
    """Verify that all models are properly set up."""
    logger.info("🔍 Verifying model setup...")
    
    required_files = [
        "models/yolov8_baseline/weights/best.pt",
        "models/fusion_model.pt",
        "models/segmentation_best.pt",
        "models/model_config.json"
    ]
    
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            logger.info(f"✅ {file_path} ({size:,} bytes)")
        else:
            logger.error(f"❌ Missing: {file_path}")
            all_good = False
    
    return all_good

def main():
    """Main function to download and set up all models."""
    logger.info("🌱 AgriSprayAI Model Downloader")
    logger.info("=" * 40)
    
    # Check if we're in the right directory
    if not Path("code/api/server.py").exists():
        logger.error("❌ Please run this script from the project root directory")
        return False
    
    logger.info("✅ Found project files")
    
    try:
        # Step 1: Create directories
        create_directories()
        
        # Step 2: Download YOLOv8 model
        if not download_yolov8_model():
            logger.error("❌ Failed to download YOLOv8 model")
            return False
        
        # Step 3: Create fusion model
        if not create_fusion_model_placeholder():
            logger.error("❌ Failed to create fusion model")
            return False
        
        # Step 4: Create segmentation model
        if not create_segmentation_model_placeholder():
            logger.error("❌ Failed to create segmentation model")
            return False
        
        # Step 5: Download pretrained models
        if not download_pretrained_models():
            logger.error("❌ Failed to download pretrained models")
            return False
        
        # Step 6: Create configuration
        create_model_config()
        
        # Step 7: Verify everything
        if verify_models():
            logger.info("\n🎉 All models downloaded and set up successfully!")
            logger.info("\n📁 Model files created:")
            logger.info("   • YOLOv8 detection model")
            logger.info("   • Multimodal fusion model")
            logger.info("   • Segmentation model")
            logger.info("   • Model configuration")
            
            logger.info("\n🚀 You can now restart the API server:")
            logger.info("   python code/api/server.py")
            
            return True
        else:
            logger.error("❌ Model verification failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during model setup: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
