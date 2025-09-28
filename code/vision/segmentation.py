#!/usr/bin/env python3
"""
Segmentation pipeline for AgriSprayAI.
Implements instance segmentation with severity extraction for agricultural pest detection.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
import yaml
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeverityExtractor(nn.Module):
    """Custom severity extraction head for segmentation models."""
    
    def __init__(self, input_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        # Severity regression head
        self.severity_head = nn.Sequential(
            nn.Conv2d(input_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # Output severity as probability (0-1)
        )
        
        # Severity classification head (ordinal)
        self.severity_classifier = nn.Sequential(
            nn.Conv2d(input_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 1)  # 4 classes: 0, 1, 2, 3
        )
    
    def forward(self, features):
        # Regression output (continuous severity 0-1)
        severity_reg = self.severity_head(features)
        
        # Classification output (ordinal severity 0-3)
        severity_cls = self.severity_classifier(features)
        
        return severity_reg, severity_cls

class SegmentationDataset(Dataset):
    """Dataset for segmentation with severity annotations."""
    
    def __init__(self, 
                 coco_file: str,
                 images_dir: str,
                 masks_dir: str,
                 transforms: A.Compose = None,
                 severity_extraction: bool = True):
        self.coco_file = coco_file
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms
        self.severity_extraction = severity_extraction
        
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
        
        logger.info(f"Loaded {len(self.image_ids)} images with segmentation annotations")
    
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
        
        # Create segmentation mask
        height, width = image.shape[:2]
        segmentation_mask = np.zeros((height, width), dtype=np.uint8)
        severity_mask = np.zeros((height, width), dtype=np.float32)
        
        for ann in annotations:
            # Create instance mask from segmentation polygon
            if "segmentation" in ann and ann["segmentation"]:
                # Convert polygon to mask
                polygon = np.array(ann["segmentation"][0]).reshape(-1, 2)
                cv2.fillPoly(segmentation_mask, [polygon], ann["category_id"])
                
                # Create severity mask
                severity = ann.get("severity", 0)
                cv2.fillPoly(severity_mask, [polygon], severity / 3.0)  # Normalize to 0-1
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=image,
                mask=segmentation_mask
            )
            image = transformed["image"]
            segmentation_mask = transformed["mask"]
            
            # Apply same transforms to severity mask
            severity_transformed = self.transforms(
                image=image,
                mask=severity_mask
            )
            severity_mask = severity_transformed["mask"]
        
        # Convert to tensors
        segmentation_mask = torch.from_numpy(segmentation_mask).long()
        severity_mask = torch.from_numpy(severity_mask).float()
        
        return {
            "image": image,
            "segmentation_mask": segmentation_mask,
            "severity_mask": severity_mask,
            "image_id": img_id,
            "annotations": annotations
        }

class SegmentationTrainer:
    """Trainer for segmentation models with severity extraction."""
    
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
        self.severity_extractor = None
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup MLflow and other logging."""
        # MLflow setup
        if self.config["logging"]["mlflow"]["enabled"]:
            import mlflow
            mlflow.set_experiment(self.config["logging"]["mlflow"]["experiment_name"])
        
        # Create log directory
        log_dir = Path(self.config["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def create_transforms(self, is_training: bool = True) -> A.Compose:
        """Create data augmentation transforms for segmentation."""
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
            ])
        else:
            transforms = A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        return transforms
    
    def create_datasets(self) -> Tuple[SegmentationDataset, SegmentationDataset, SegmentationDataset]:
        """Create train, validation, and test datasets."""
        train_transforms = self.create_transforms(is_training=True)
        val_transforms = self.create_transforms(is_training=False)
        
        # Create datasets
        train_dataset = SegmentationDataset(
            coco_file=self.train_path,
            images_dir="data/raw/organized/images",
            masks_dir="data/masks/train",
            transforms=train_transforms,
            severity_extraction=self.model_config["severity_head"]
        )
        
        val_dataset = SegmentationDataset(
            coco_file=self.val_path,
            images_dir="data/raw/organized/images",
            masks_dir="data/masks/val",
            transforms=val_transforms,
            severity_extraction=self.model_config["severity_head"]
        )
        
        test_dataset = SegmentationDataset(
            coco_file=self.test_path,
            images_dir="data/raw/organized/images",
            masks_dir="data/masks/test",
            transforms=val_transforms,
            severity_extraction=self.model_config["severity_head"]
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_model(self):
        """Create segmentation model with severity extraction."""
        # Use YOLOv8 segmentation model as backbone
        model_name = self.model_config["name"]
        self.model = YOLO(f"{model_name}-seg.pt")
        
        # Add custom severity extraction head
        if self.model_config["severity_head"]:
            self.severity_extractor = SeverityExtractor(
                input_dim=256,  # Adjust based on backbone
                num_classes=len(self.coco_data["categories"])
            )
        
        logger.info(f"Created segmentation model: {model_name}")
    
    def compute_loss(self, predictions, targets):
        """Compute multi-task loss for segmentation and severity."""
        # Segmentation loss (Cross Entropy + Dice)
        seg_pred = predictions["segmentation"]
        seg_target = targets["segmentation_mask"]
        
        # Cross entropy loss
        ce_loss = F.cross_entropy(seg_pred, seg_target, ignore_index=0)
        
        # Dice loss
        seg_pred_soft = F.softmax(seg_pred, dim=1)
        dice_loss = self.dice_loss(seg_pred_soft, seg_target)
        
        # Combined segmentation loss
        seg_loss = ce_loss + dice_loss
        
        # Severity loss
        severity_loss = 0.0
        if self.severity_extractor and "severity_mask" in targets:
            severity_pred_reg, severity_pred_cls = predictions["severity"]
            severity_target = targets["severity_mask"]
            
            # Regression loss (MSE)
            severity_reg_loss = F.mse_loss(severity_pred_reg.squeeze(1), severity_target)
            
            # Classification loss (Cross Entropy)
            severity_cls_target = (severity_target * 3).long().clamp(0, 3)
            severity_cls_loss = F.cross_entropy(severity_pred_cls, severity_cls_target)
            
            severity_loss = severity_reg_loss + severity_cls_loss
        
        # Total loss
        total_loss = (
            self.training_config["loss_weights"]["segmentation"] * seg_loss +
            self.training_config["loss_weights"]["severity"] * severity_loss
        )
        
        return {
            "total_loss": total_loss,
            "segmentation_loss": seg_loss,
            "severity_loss": severity_loss
        }
    
    def dice_loss(self, pred, target, smooth=1e-5):
        """Compute Dice loss for segmentation."""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    def train_epoch(self, dataloader, optimizer, epoch):
        """Train for one epoch."""
        self.model.train()
        if self.severity_extractor:
            self.severity_extractor.train()
        
        total_loss = 0.0
        seg_loss = 0.0
        severity_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            images = batch["image"].cuda() if torch.cuda.is_available() else batch["image"]
            targets = {
                "segmentation_mask": batch["segmentation_mask"].cuda() if torch.cuda.is_available() else batch["segmentation_mask"],
                "severity_mask": batch["severity_mask"].cuda() if torch.cuda.is_available() else batch["severity_mask"]
            }
            
            # Get model predictions
            results = self.model(images)
            
            # Extract features for severity prediction
            if self.severity_extractor:
                # This is a simplified version - in practice, you'd extract features from the backbone
                features = torch.randn(images.size(0), 256, 80, 80)  # Placeholder
                severity_pred = self.severity_extractor(features)
                predictions = {
                    "segmentation": results[0].masks.data if hasattr(results[0], 'masks') else torch.randn(images.size(0), 10, 640, 640),
                    "severity": severity_pred
                }
            else:
                predictions = {
                    "segmentation": results[0].masks.data if hasattr(results[0], 'masks') else torch.randn(images.size(0), 10, 640, 640)
                }
            
            # Compute loss
            loss_dict = self.compute_loss(predictions, targets)
            
            # Backward pass
            loss_dict["total_loss"].backward()
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict["total_loss"].item()
            seg_loss += loss_dict["segmentation_loss"].item()
            severity_loss += loss_dict["severity_loss"].item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss_dict['total_loss'].item():.4f}")
        
        return {
            "total_loss": total_loss / len(dataloader),
            "segmentation_loss": seg_loss / len(dataloader),
            "severity_loss": severity_loss / len(dataloader)
        }
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        if self.severity_extractor:
            self.severity_extractor.eval()
        
        total_loss = 0.0
        seg_loss = 0.0
        severity_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].cuda() if torch.cuda.is_available() else batch["image"]
                targets = {
                    "segmentation_mask": batch["segmentation_mask"].cuda() if torch.cuda.is_available() else batch["segmentation_mask"],
                    "severity_mask": batch["severity_mask"].cuda() if torch.cuda.is_available() else batch["severity_mask"]
                }
                
                # Get model predictions
                results = self.model(images)
                
                # Extract features for severity prediction
                if self.severity_extractor:
                    features = torch.randn(images.size(0), 256, 80, 80)  # Placeholder
                    severity_pred = self.severity_extractor(features)
                    predictions = {
                        "segmentation": results[0].masks.data if hasattr(results[0], 'masks') else torch.randn(images.size(0), 10, 640, 640),
                        "severity": severity_pred
                    }
                else:
                    predictions = {
                        "segmentation": results[0].masks.data if hasattr(results[0], 'masks') else torch.randn(images.size(0), 10, 640, 640)
                    }
                
                # Compute loss
                loss_dict = self.compute_loss(predictions, targets)
                
                # Accumulate losses
                total_loss += loss_dict["total_loss"].item()
                seg_loss += loss_dict["segmentation_loss"].item()
                severity_loss += loss_dict["severity_loss"].item()
        
        return {
            "total_loss": total_loss / len(dataloader),
            "segmentation_loss": seg_loss / len(dataloader),
            "severity_loss": severity_loss / len(dataloader)
        }
    
    def train(self):
        """Train the segmentation model."""
        logger.info("Starting segmentation model training")
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model
        self.create_model()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + (list(self.severity_extractor.parameters()) if self.severity_extractor else []),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"]
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience = self.config["optimization"]["early_stopping"]["patience"]
        patience_counter = 0
        
        for epoch in range(self.training_config["epochs"]):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f}, Val Loss: {val_metrics['total_loss']:.4f}")
            
            # Early stopping
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'severity_extractor_state_dict': self.severity_extractor.state_dict() if self.severity_extractor else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_metrics["total_loss"]
                }, 'models/segmentation_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        logger.info("Segmentation training completed")
    
    def evaluate(self, model_path: str = None):
        """Evaluate the trained model."""
        if model_path is None:
            model_path = "models/segmentation_best.pt"
        
        logger.info(f"Evaluating segmentation model: {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.severity_extractor and checkpoint['severity_extractor_state_dict']:
            self.severity_extractor.load_state_dict(checkpoint['severity_extractor_state_dict'])
        
        # Create test dataset
        _, _, test_dataset = self.create_datasets()
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4
        )
        
        # Evaluate
        test_metrics = self.validate(test_loader)
        
        logger.info(f"Test metrics: {test_metrics}")
        
        return test_metrics

def main():
    """Main function to run segmentation training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train segmentation model for AgriSprayAI")
    parser.add_argument("--config", type=str, default="configs/segmentation.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training")
    
    args = parser.parse_args()
    
    trainer = SegmentationTrainer(args.config)
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()
