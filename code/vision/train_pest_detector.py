#!/usr/bin/env python3
"""
Train YOLOv8 detector specifically for the agricultural pests dataset.
Optimized for the 12-class pest detection with severity prediction.
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.pytorch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PestDetectorTrainer:
    """Trainer specifically for agricultural pest detection."""
    
    def __init__(self, config_path: str = "configs/yolov8_baseline.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config["model"]
        self.training_config = self.config["training"]
        self.data_config = self.config["data"]
        
        # Pest categories from the dataset
        self.pest_categories = [
            'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
            'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
        ]
        
        # Severity mapping
        self.severity_mapping = {
            'ants': 1, 'bees': 0, 'beetle': 2, 'catterpillar': 3, 'earthworms': 0,
            'earwig': 2, 'grasshopper': 3, 'moth': 2, 'slug': 3, 'snail': 3,
            'wasp': 1, 'weevil': 3
        }
        
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
    
    def create_yaml_config(self) -> str:
        """Create YOLOv8 YAML configuration file."""
        yaml_config = {
            'path': str(Path(self.data_config["images_dir"]).parent.absolute()),
            'train': 'annotated/instances_train.json',
            'val': 'annotated/instances_val.json',
            'test': 'annotated/instances_test.json',
            'nc': len(self.pest_categories),
            'names': self.pest_categories
        }
        
        yaml_file = Path("configs/pest_dataset.yaml")
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        logger.info(f"Created YOLOv8 config: {yaml_file}")
        return str(yaml_file)
    
    def train_model(self):
        """Train the YOLOv8 model on pest dataset."""
        logger.info("Starting YOLOv8 pest detection training")
        
        # Create YAML config
        yaml_config = self.create_yaml_config()
        
        # Initialize model
        model_name = self.model_config["name"]
        self.model = YOLO(f"{model_name}.pt")
        
        # Train the model
        results = self.model.train(
            data=yaml_config,
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
            name="pest_detector",
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
        
        logger.info("Pest detection training completed")
        return results
    
    def evaluate_model(self, model_path: str = None):
        """Evaluate the trained model on test set."""
        if model_path is None:
            model_path = "models/pest_detector/weights/best.pt"
        
        logger.info(f"Evaluating pest detection model: {model_path}")
        
        # Load the trained model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data="configs/pest_dataset.yaml",
            imgsz=self.model_config["input_size"][0],
            batch=16,
            conf=0.5,
            iou=0.45,
            max_det=300,
            save_json=True,
            save_hybrid=True,
            plots=True
        )
        
        # Extract metrics
        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
            "f1": results.box.f1
        }
        
        # Per-class metrics
        if hasattr(results.box, 'maps'):
            per_class_metrics = {}
            for i, category in enumerate(self.pest_categories):
                per_class_metrics[category] = {
                    "mAP50": results.box.maps[i] if i < len(results.box.maps) else 0.0,
                    "severity": self.severity_mapping[category]
                }
            metrics["per_class"] = per_class_metrics
        
        logger.info(f"Validation metrics: {metrics}")
        
        # Log to MLflow if enabled
        if self.config["logging"]["mlflow"]["enabled"]:
            with mlflow.start_run():
                mlflow.log_metrics(metrics)
                mlflow.pytorch.log_model(model, "model")
        
        return results, metrics
    
    def analyze_predictions(self, model_path: str = None):
        """Analyze model predictions and create visualizations."""
        if model_path is None:
            model_path = "models/pest_detector/weights/best.pt"
        
        logger.info("Analyzing model predictions...")
        
        # Load model
        model = YOLO(model_path)
        
        # Load test data
        test_file = Path(self.data_config["test_path"])
        if not test_file.exists():
            logger.warning("Test data not found, skipping analysis")
            return
        
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        # Analyze predictions
        predictions = []
        ground_truth = []
        
        for img_info in test_data["images"][:100]:  # Analyze first 100 images
            img_path = Path(self.data_config["images_dir"]) / img_info["file_name"]
            
            if not img_path.exists():
                continue
            
            # Get predictions
            results = model(str(img_path))
            
            # Get ground truth
            img_annotations = [ann for ann in test_data["annotations"] 
                             if ann["image_id"] == img_info["id"]]
            
            for ann in img_annotations:
                ground_truth.append(ann["category_id"] - 1)  # Convert to 0-based
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        pred_class = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        predictions.append((pred_class, confidence))
        
        # Create confusion matrix
        if predictions and ground_truth:
            pred_classes = [pred[0] for pred in predictions]
            
            # Pad or truncate to match lengths
            min_len = min(len(pred_classes), len(ground_truth))
            pred_classes = pred_classes[:min_len]
            ground_truth = ground_truth[:min_len]
            
            cm = confusion_matrix(ground_truth, pred_classes)
            
            # Plot confusion matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.pest_categories,
                       yticklabels=self.pest_categories)
            plt.title('Confusion Matrix - Pest Detection')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('models/pest_detector/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Classification report
            report = classification_report(ground_truth, pred_classes,
                                         target_names=self.pest_categories,
                                         output_dict=True)
            
            # Save report
            with open('models/pest_detector/classification_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("Analysis complete - saved confusion matrix and classification report")
    
    def create_severity_analysis(self, metrics: Dict[str, Any]):
        """Create severity-based analysis of model performance."""
        logger.info("Creating severity-based analysis...")
        
        if "per_class" not in metrics:
            logger.warning("Per-class metrics not available")
            return
        
        # Group by severity
        severity_groups = {0: [], 1: [], 2: [], 3: []}
        
        for category, class_metrics in metrics["per_class"].items():
            severity = class_metrics["severity"]
            mAP50 = class_metrics["mAP50"]
            severity_groups[severity].append((category, mAP50))
        
        # Calculate average performance by severity
        severity_performance = {}
        for severity, categories in severity_groups.items():
            if categories:
                avg_mAP = sum(mAP for _, mAP in categories) / len(categories)
                severity_performance[severity] = {
                    "average_mAP50": avg_mAP,
                    "categories": [cat for cat, _ in categories],
                    "count": len(categories)
                }
        
        # Create visualization
        severities = list(severity_performance.keys())
        avg_maps = [severity_performance[s]["average_mAP50"] for s in severities]
        severity_names = ['Beneficial', 'Low', 'Medium', 'High']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(severity_names, avg_maps, color=['green', 'yellow', 'orange', 'red'])
        plt.title('Model Performance by Pest Severity Level')
        plt.xlabel('Severity Level')
        plt.ylabel('Average mAP@0.5')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_maps):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('models/pest_detector/severity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save analysis
        analysis = {
            "severity_performance": severity_performance,
            "summary": {
                "beneficial_organisms": severity_performance.get(0, {}).get("average_mAP50", 0),
                "low_severity_pests": severity_performance.get(1, {}).get("average_mAP50", 0),
                "medium_severity_pests": severity_performance.get(2, {}).get("average_mAP50", 0),
                "high_severity_pests": severity_performance.get(3, {}).get("average_mAP50", 0)
            }
        }
        
        with open('models/pest_detector/severity_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info("Severity analysis complete")
    
    def export_model(self, model_path: str = None):
        """Export model for edge deployment."""
        if model_path is None:
            model_path = "models/pest_detector/weights/best.pt"
        
        logger.info(f"Exporting pest detection model: {model_path}")
        
        # Load the trained model
        model = YOLO(model_path)
        
        # Export to different formats
        export_formats = self.config["export"]["formats"]
        
        for format_type in export_formats:
            try:
                if format_type == "onnx":
                    model.export(format="onnx", dynamic=True, simplify=True)
                    logger.info("Exported to ONNX format")
                elif format_type == "tensorrt":
                    model.export(format="engine", device=0)
                    logger.info("Exported to TensorRT format")
                elif format_type == "tflite":
                    model.export(format="tflite", int8=True)
                    logger.info("Exported to TFLite format")
                
            except Exception as e:
                logger.error(f"Failed to export to {format_type}: {e}")
    
    def run(self):
        """Run the complete training pipeline."""
        try:
            logger.info("Starting pest detection training pipeline")
            
            # Step 1: Train model
            training_results = self.train_model()
            
            # Step 2: Evaluate model
            eval_results, metrics = self.evaluate_model()
            
            # Step 3: Analyze predictions
            self.analyze_predictions()
            
            # Step 4: Create severity analysis
            self.create_severity_analysis(metrics)
            
            # Step 5: Export model
            self.export_model()
            
            logger.info("Pest detection training pipeline completed successfully")
            
            # Print summary
            print("\n" + "="*60)
            print("PEST DETECTION TRAINING SUMMARY")
            print("="*60)
            print(f"Model: {self.model_config['name']}")
            print(f"Classes: {len(self.pest_categories)} pest categories")
            print(f"Training Epochs: {self.training_config['epochs']}")
            print(f"Batch Size: {self.training_config['batch_size']}")
            print(f"\nPerformance Metrics:")
            print(f"  mAP@0.5: {metrics['mAP50']:.3f}")
            print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1']:.3f}")
            
            if "per_class" in metrics:
                print(f"\nPer-Class Performance (Top 5):")
                sorted_classes = sorted(metrics["per_class"].items(), 
                                      key=lambda x: x[1]["mAP50"], reverse=True)
                for category, class_metrics in sorted_classes[:5]:
                    severity = class_metrics["severity"]
                    mAP = class_metrics["mAP50"]
                    print(f"  {category}: {mAP:.3f} (Severity: {severity})")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main function to run the pest detector trainer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 pest detector for AgriSprayAI")
    parser.add_argument("--config", type=str, default="configs/yolov8_baseline.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training")
    
    args = parser.parse_args()
    
    trainer = PestDetectorTrainer(args.config)
    trainer.run()

if __name__ == "__main__":
    main()
