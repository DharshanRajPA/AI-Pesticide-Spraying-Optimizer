#!/usr/bin/env python3
"""
Multimodal fusion model for AgriSprayAI.
Implements both concatenation and cross-attention fusion for vision and text features.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass
from enum import Enum
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FusionType(Enum):
    """Fusion method enumeration."""
    CONCATENATION = "concatenation"
    CROSS_ATTENTION = "cross_attention"

@dataclass
class FusionConfig:
    """Configuration for fusion model."""
    fusion_type: FusionType
    vision_dim: int
    text_dim: int
    hidden_dims: List[int]
    num_classes: int
    dropout: float
    activation: str

class ConcatenationFusion(nn.Module):
    """Simple concatenation-based fusion for edge deployment."""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Input dimensions
        input_dim = config.vision_dim + config.text_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.GELU(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim
        
        self.fusion_mlp = nn.Sequential(*layers)
        
        # Output heads
        self.classification_head = nn.Sequential(
            nn.Linear(prev_dim, config.num_classes),
            nn.Softmax(dim=-1)
        )
        
        self.severity_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for concatenation fusion."""
        # Concatenate features
        fused_features = torch.cat([vision_features, text_features], dim=-1)
        
        # Pass through MLP
        fused_features = self.fusion_mlp(fused_features)
        
        # Generate outputs
        classification = self.classification_head(fused_features)
        severity = self.severity_head(fused_features)
        confidence = self.confidence_head(fused_features)
        
        return {
            "classification": classification,
            "severity": severity,
            "confidence": confidence,
            "fused_features": fused_features
        }

class CrossAttentionFusion(nn.Module):
    """Cross-attention based fusion for cloud deployment."""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Projection layers
        self.vision_proj = nn.Linear(config.vision_dim, config.hidden_dims[0])
        self.text_proj = nn.Linear(config.text_dim, config.hidden_dims[0])
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dims[0],
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dims[0])
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dims[0], config.hidden_dims[0] * 4),
            nn.ReLU() if config.activation == "relu" else nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0] * 4, config.hidden_dims[0]),
            nn.Dropout(config.dropout)
        )
        
        # Output heads
        self.classification_head = nn.Sequential(
            nn.Linear(config.hidden_dims[0], config.num_classes),
            nn.Softmax(dim=-1)
        )
        
        self.severity_head = nn.Sequential(
            nn.Linear(config.hidden_dims[0], 1),
            nn.Sigmoid()
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dims[0], 1),
            nn.Sigmoid()
        )
    
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for cross-attention fusion."""
        # Project features to same dimension
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        # Add sequence dimension for attention
        vision_proj = vision_proj.unsqueeze(1)  # [batch, 1, hidden_dim]
        text_proj = text_proj.unsqueeze(1)      # [batch, 1, hidden_dim]
        
        # Cross-attention: vision attends to text
        attended_features, attention_weights = self.cross_attention(
            query=vision_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Residual connection and layer norm
        fused_features = self.layer_norm(vision_proj + attended_features)
        
        # Feed-forward network
        fused_features = fused_features + self.ffn(fused_features)
        
        # Remove sequence dimension
        fused_features = fused_features.squeeze(1)
        
        # Generate outputs
        classification = self.classification_head(fused_features)
        severity = self.severity_head(fused_features)
        confidence = self.confidence_head(fused_features)
        
        return {
            "classification": classification,
            "severity": severity,
            "confidence": confidence,
            "fused_features": fused_features,
            "attention_weights": attention_weights
        }

class MultimodalFusionModel(nn.Module):
    """Main multimodal fusion model."""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Choose fusion method
        if config.fusion_type == FusionType.CONCATENATION:
            self.fusion = ConcatenationFusion(config)
        elif config.fusion_type == FusionType.CROSS_ATTENTION:
            self.fusion = CrossAttentionFusion(config)
        else:
            raise ValueError(f"Unknown fusion type: {config.fusion_type}")
    
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.fusion(vision_features, text_features)

class FusionDataset(Dataset):
    """Dataset for multimodal fusion training."""
    
    def __init__(self, 
                 vision_features_path: str,
                 text_features_path: str,
                 labels_path: str,
                 severity_path: str = None):
        self.vision_features_path = Path(vision_features_path)
        self.text_features_path = Path(text_features_path)
        self.labels_path = Path(labels_path)
        self.severity_path = Path(severity_path) if severity_path else None
        
        # Load data
        self.vision_features = np.load(self.vision_features_path)
        self.text_features = np.load(self.text_features_path)
        self.labels = np.load(self.labels_path)
        self.severity = np.load(self.severity_path) if self.severity_path else None
        
        # Validate data consistency
        assert len(self.vision_features) == len(self.text_features) == len(self.labels)
        if self.severity is not None:
            assert len(self.severity) == len(self.labels)
        
        logger.info(f"Loaded fusion dataset: {len(self.vision_features)} samples")
    
    def __len__(self):
        return len(self.vision_features)
    
    def __getitem__(self, idx):
        item = {
            "vision_features": torch.tensor(self.vision_features[idx], dtype=torch.float32),
            "text_features": torch.tensor(self.text_features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        if self.severity is not None:
            item["severity"] = torch.tensor(self.severity[idx], dtype=torch.float32)
        
        return item

class FusionTrainer:
    """Trainer for multimodal fusion model."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config["model"]
        self.training_config = self.config["training"]
        self.data_config = self.config["data"]
        
        # Create fusion config
        self.fusion_config = FusionConfig(
            fusion_type=FusionType(self.model_config["fusion"]["type"]),
            vision_dim=self.model_config["vision"]["feature_dim"],
            text_dim=self.model_config["text"]["feature_dim"],
            hidden_dims=self.model_config["fusion"]["hidden_dims"],
            num_classes=self.model_config["heads"]["classification"]["num_classes"],
            dropout=self.model_config["fusion"]["dropout"],
            activation=self.model_config["fusion"]["activation"]
        )
        
        # Initialize model
        self.model = MultimodalFusionModel(self.fusion_config)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for training."""
        log_dir = Path(self.config["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def create_datasets(self) -> Tuple[FusionDataset, FusionDataset, FusionDataset]:
        """Create train, validation, and test datasets."""
        train_dataset = FusionDataset(
            vision_features_path=self.data_config["vision"]["train_path"],
            text_features_path=self.data_config["text"]["train_path"],
            labels_path=self.data_config["fusion"]["train_path"],
            severity_path=self.data_config.get("severity", {}).get("train_path")
        )
        
        val_dataset = FusionDataset(
            vision_features_path=self.data_config["vision"]["val_path"],
            text_features_path=self.data_config["text"]["val_path"],
            labels_path=self.data_config["fusion"]["val_path"],
            severity_path=self.data_config.get("severity", {}).get("val_path")
        )
        
        test_dataset = FusionDataset(
            vision_features_path=self.data_config["vision"]["test_path"],
            text_features_path=self.data_config["text"]["test_path"],
            labels_path=self.data_config["fusion"]["test_path"],
            severity_path=self.data_config.get("severity", {}).get("test_path")
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss."""
        losses = {}
        
        # Classification loss
        classification_loss = F.cross_entropy(
            predictions["classification"], 
            targets["labels"]
        )
        losses["classification"] = classification_loss
        
        # Severity loss (if available)
        if "severity" in targets and "severity" in predictions:
            severity_loss = F.mse_loss(
                predictions["severity"].squeeze(),
                targets["severity"]
            )
            losses["severity"] = severity_loss
        else:
            losses["severity"] = torch.tensor(0.0, device=predictions["classification"].device)
        
        # Confidence loss (self-supervised)
        confidence_loss = F.binary_cross_entropy(
            predictions["confidence"].squeeze(),
            torch.ones_like(predictions["confidence"].squeeze())  # Target high confidence
        )
        losses["confidence"] = confidence_loss
        
        # Total loss
        total_loss = (
            self.training_config["loss_weights"]["classification"] * classification_loss +
            self.training_config["loss_weights"]["severity"] * losses["severity"] +
            self.training_config["loss_weights"]["confidence"] * confidence_loss
        )
        losses["total"] = total_loss
        
        return losses
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_losses = {"total": 0.0, "classification": 0.0, "severity": 0.0, "confidence": 0.0}
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Move to device
            device = next(self.model.parameters()).device
            vision_features = batch["vision_features"].to(device)
            text_features = batch["text_features"].to(device)
            labels = batch["labels"].to(device)
            
            targets = {"labels": labels}
            if "severity" in batch:
                targets["severity"] = batch["severity"].to(device)
            
            # Forward pass
            predictions = self.model(vision_features, text_features)
            
            # Compute loss
            losses = self.compute_loss(predictions, targets)
            
            # Backward pass
            losses["total"].backward()
            optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                total_losses[key] += value.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {losses['total'].item():.4f}")
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        
        return total_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_losses = {"total": 0.0, "classification": 0.0, "severity": 0.0, "confidence": 0.0}
        all_predictions = []
        all_labels = []
        all_severity_pred = []
        all_severity_true = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                device = next(self.model.parameters()).device
                vision_features = batch["vision_features"].to(device)
                text_features = batch["text_features"].to(device)
                labels = batch["labels"].to(device)
                
                targets = {"labels": labels}
                if "severity" in batch:
                    targets["severity"] = batch["severity"].to(device)
                
                # Forward pass
                predictions = self.model(vision_features, text_features)
                
                # Compute loss
                losses = self.compute_loss(predictions, targets)
                
                # Accumulate losses
                for key, value in losses.items():
                    total_losses[key] += value.item()
                
                # Collect predictions for metrics
                all_predictions.extend(predictions["classification"].argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if "severity" in predictions and "severity" in targets:
                    all_severity_pred.extend(predictions["severity"].squeeze().cpu().numpy())
                    all_severity_true.extend(targets["severity"].cpu().numpy())
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        if all_severity_pred:
            severity_mae = np.mean(np.abs(np.array(all_severity_pred) - np.array(all_severity_true)))
            metrics["severity_mae"] = severity_mae
        
        return {**total_losses, **metrics}
    
    def train(self):
        """Train the fusion model."""
        logger.info("Starting multimodal fusion training")
        
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
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"]
        )
        
        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_config["epochs"]
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
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total']:.4f}, Val Loss: {val_metrics['total']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_metrics': val_metrics
                }, 'models/fusion_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        logger.info("Fusion training completed")
    
    def evaluate(self, model_path: str = None) -> Dict[str, float]:
        """Evaluate the trained model."""
        if model_path is None:
            model_path = "models/fusion_best.pt"
        
        logger.info(f"Evaluating fusion model: {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create test dataset
        _, _, test_dataset = self.create_datasets()
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        # Evaluate
        test_metrics = self.validate(test_loader)
        
        logger.info(f"Test metrics: {test_metrics}")
        
        return test_metrics

def main():
    """Main function to run fusion training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train multimodal fusion model for AgriSprayAI")
    parser.add_argument("--config", type=str, default="configs/fusion_model.yaml",
                       help="Path to configuration file")
    parser.add_argument("--fusion-type", type=str, choices=["concatenation", "cross_attention"],
                       default="concatenation", help="Fusion method to use")
    
    args = parser.parse_args()
    
    trainer = FusionTrainer(args.config)
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()
