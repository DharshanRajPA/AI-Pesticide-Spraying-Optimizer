#!/usr/bin/env python3
"""
Train YOLOv8 detector for agricultural pest detection with severity prediction.
This script implements the baseline detector training pipeline for AgriSprayAI.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriSprayDataset(Dataset):
    """Custom dataset for agricultural pest detection with severity."""
    
    def __init__(self, 
                 coco_file: str,
                 images_dir: str,
                 transforms: A.Compose = None,
                 severity_head: bool = True):
        self.coco_file = coco_file
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.severity_head = severity_head
        
        # Load COCO annotations
        with open(coco_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.image_id_to_info = {img["id"]: img for img in self.coco_data["images"]}
        self.category_id_to_name = {cat["id"]: cat["name"] for cat in self.coco_data["categories"]}
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # Filter images with annotations
        self.image_ids = [img_id for img_id in self.image_annotations.keys()]
        
        logger.info(f"Loaded {len(self.image_ids)} images with annotations")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.image_id_to_info[img_id]
        annotations = self.image_annotations[img_id]
        
        # Load image
        img_path = self.images_dir / img_info["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare annotations
        boxes = []
        labels = []
        severities = []
        
        for ann in annotations:
            # Bounding box [x, y, width, height] -> [x1, y1, x2, y2]
            bbox = ann["bbox"]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"] - 1)  # Convert to 0-based indexing
            severities.append(ann.get("severity", 0))
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        severities = torch.tensor(severities, dtype=torch.float32)
        
        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "severities": severities,
            "image_id": img_id
        }

class SeverityHead(nn.Module):
    """Custom severity prediction head for YOLOv8."""
    
    def __init__(self, input_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        # Severity regression head
        self.severity_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output severity as probability
        )
    
    def forward(self, features):
        return self.severity_head(features)

class YOLOv8Trainer:
    """YOLOv8 trainer with custom severity head."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config["model"]
        self.training_config = self.config["training"]
        self.data_config = self.config["data"]
        
        # Setup paths
        self.train_path = self.data_config["train_path"]
        self.val_path = self.data_config["val_path"]
        self.test_path = self.data_config["test_path"]
        
        # Setup model
        self.model = None
        self.severity_head = None
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup MLflow and other logging."""
        # MLflow setup
        if self.config["logging"]["mlflow"]["enabled"]:
            mlflow.set_experiment(self.config["logging"]["mlflow"]["experiment_name"])
        
        # Create log directory
        log_dir = Path(self.config["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def create_transforms(self, is_training: bool = True) -> A.Compose:
        """Create data augmentation transforms."""
        if is_training:
            transforms = A.Compose([
                A.Resize(640, 640),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            transforms = A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        return transforms
    
    def create_datasets(self) -> Tuple[AgriSprayDataset, AgriSprayDataset, AgriSprayDataset]:
        """Create train, validation, and test datasets."""
        train_transforms = self.create_transforms(is_training=True)
        val_transforms = self.create_transforms(is_training=False)
        
        # Create datasets
        train_dataset = AgriSprayDataset(
            coco_file=self.train_path,
            images_dir="data/raw/organized/images",
            transforms=train_transforms,
            severity_head=self.model_config["severity_head"]
        )
        
        val_dataset = AgriSprayDataset(
            coco_file=self.val_path,
            images_dir="data/raw/organized/images",
            transforms=val_transforms,
            severity_head=self.model_config["severity_head"]
        )
        
        test_dataset = AgriSprayDataset(
            coco_file=self.test_path,
            images_dir="data/raw/organized/images",
            transforms=val_transforms,
            severity_head=self.model_config["severity_head"]
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(self):
        """Train the YOLOv8 model."""
        logger.info("Starting YOLOv8 training")
        
        # Initialize model
        model_name = self.model_config["name"]
        self.model = YOLO(f"{model_name}.pt")
        
        # Train the model
        results = self.model.train(
            data={
                "train": self.train_path,
                "val": self.val_path,
                "test": self.test_path
            },
            epochs=self.training_config["epochs"],
            batch=self.training_config["batch_size"],
            imgsz=self.model_config["input_size"][0],
            lr0=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
            momentum=self.training_config["momentum"],
            warmup_epochs=self.training_config["warmup_epochs"],
            warmup_momentum=self.training_config["warmup_momentum"],
            warmup_bias_lr=self.training_config["warmup_bias_lr"],
            hsv_h=self.data_config["augmentation"]["hsv_h"],
            hsv_s=self.data_config["augmentation"]["hsv_s"],
            hsv_v=self.data_config["augmentation"]["hsv_v"],
            degrees=self.data_config["augmentation"]["degrees"],
            translate=self.data_config["augmentation"]["translate"],
            scale=self.data_config["augmentation"]["scale"],
            shear=self.data_config["augmentation"]["shear"],
            perspective=self.data_config["augmentation"]["perspective"],
            flipud=self.data_config["augmentation"]["flipud"],
            fliplr=self.data_config["augmentation"]["fliplr"],
            mosaic=self.data_config["augmentation"]["mosaic"],
            mixup=self.data_config["augmentation"]["mixup"],
            project="models",
            name="yolov8_baseline",
            exist_ok=True,
            save=True,
            save_period=10,
            cache=True,
            device=self.config["hardware"]["device"],
            workers=self.config["hardware"]["num_workers"],
            patience=self.config["optimization"]["early_stopping"]["patience"],
            save_json=True,
            verbose=True
        )
        
        logger.info("Training completed")
        return results
    
    def evaluate_model(self, model_path: str = None):
        """Evaluate the trained model."""
        if model_path is None:
            model_path = "models/yolov8_baseline/weights/best.pt"
        
        logger.info(f"Evaluating model: {model_path}")
        
        # Load the trained model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data={
                "val": self.val_path,
                "test": self.test_path
            },
            imgsz=self.model_config["input_size"][0],
            batch=16,
            conf=0.5,
            iou=0.45,
            max_det=300,
            save_json=True,
            save_hybrid=True,
            plots=True
        )
        
        # Log metrics
        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
            "f1": results.box.f1
        }
        
        logger.info(f"Validation metrics: {metrics}")
        
        # Log to MLflow if enabled
        if self.config["logging"]["mlflow"]["enabled"]:
            with mlflow.start_run():
                mlflow.log_metrics(metrics)
                mlflow.pytorch.log_model(model, "model")
        
        return results, metrics
    
    def export_model(self, model_path: str = None):
        """Export model for edge deployment."""
        if model_path is None:
            model_path = "models/yolov8_baseline/weights/best.pt"
        
        logger.info(f"Exporting model: {model_path}")
        
        # Load the trained model
        model = YOLO(model_path)
        
        # Export to different formats
        export_formats = self.config["export"]["formats"]
        
        for format_type in export_formats:
            try:
                if format_type == "onnx":
                    model.export(format="onnx", dynamic=True, simplify=True)
                elif format_type == "tensorrt":
                    model.export(format="engine", device=0)
                elif format_type == "tflite":
                    model.export(format="tflite", int8=True)
                
                logger.info(f"Exported to {format_type} format")
                
            except Exception as e:
                logger.error(f"Failed to export to {format_type}: {e}")
    
    def run(self):
        """Run the complete training pipeline."""
        try:
            # Create datasets
            train_dataset, val_dataset, test_dataset = self.create_datasets()
            logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
            
            # Train model
            training_results = self.train_model()
            
            # Evaluate model
            eval_results, metrics = self.evaluate_model()
            
            # Export model
            self.export_model()
            
            logger.info("Training pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main function to run the trainer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector for AgriSprayAI")
    parser.add_argument("--config", type=str, default="configs/yolov8_baseline.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training")
    
    args = parser.parse_args()
    
    trainer = YOLOv8Trainer(args.config)
    trainer.run()

if __name__ == "__main__":
    main()
