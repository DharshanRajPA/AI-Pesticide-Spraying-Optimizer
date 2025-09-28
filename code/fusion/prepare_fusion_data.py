#!/usr/bin/env python3
"""
Prepare fusion data for multimodal training.
Extracts features from vision and text models for fusion training.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
import pickle

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from vision.train_detector import AgriSprayDataset
from nlp.gemini_nlp_pipeline import GeminiASRNLPPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from vision and text models for fusion training."""
    
    def __init__(self, config_path: str = "configs/fusion_model.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize models
        self.vision_model = None
        self.nlp_pipeline = None
        
        self.setup_models()
    
    def setup_models(self):
        """Setup vision and NLP models for feature extraction."""
        try:
            # Load vision model
            from ultralytics import YOLO
            vision_model_path = self.config["models"]["vision"]["model_path"]
            if os.path.exists(vision_model_path):
                self.vision_model = YOLO(vision_model_path)
                logger.info(f"Loaded vision model: {vision_model_path}")
            else:
                logger.warning(f"Vision model not found: {vision_model_path}")
            
            # Initialize NLP pipeline
            self.nlp_pipeline = GeminiASRNLPPipeline()
            logger.info("Initialized Gemini NLP pipeline")
            
        except Exception as e:
            logger.error(f"Failed to setup models: {e}")
            raise
    
    def extract_vision_features(self, image_path: str) -> np.ndarray:
        """Extract vision features from image."""
        try:
            if self.vision_model is None:
                # Return random features if model not available
                return np.random.randn(512).astype(np.float32)
            
            # Run inference
            results = self.vision_model(image_path)
            
            # Extract features from the last layer
            # This is a simplified version - in practice, you'd extract from intermediate layers
            if results and len(results) > 0:
                # Get the feature map from the last layer
                # For YOLOv8, we need to modify the model to extract features
                # This is a placeholder implementation
                features = np.random.randn(512).astype(np.float32)
            else:
                features = np.zeros(512, dtype=np.float32)
            
            return features
            
        except Exception as e:
            logger.error(f"Vision feature extraction failed: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """Extract text features from text."""
        try:
            if self.nlp_pipeline is None:
                # Return random features if pipeline not available
                return np.random.randn(384).astype(np.float32)
            
            # Process text through NLP pipeline
            result = self.nlp_pipeline.process_text(text)
            
            if result["success"] and result["text_embedding"]:
                return np.array(result["text_embedding"].embedding, dtype=np.float32)
            else:
                return np.zeros(384, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Text feature extraction failed: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def extract_features_batch(self, 
                              image_paths: List[str], 
                              texts: List[str],
                              labels: List[int],
                              severities: List[int] = None) -> Dict[str, np.ndarray]:
        """Extract features for a batch of samples."""
        
        vision_features = []
        text_features = []
        
        logger.info(f"Extracting features for {len(image_paths)} samples")
        
        for i, (image_path, text) in enumerate(tqdm(zip(image_paths, texts), total=len(image_paths))):
            # Extract vision features
            vision_feat = self.extract_vision_features(image_path)
            vision_features.append(vision_feat)
            
            # Extract text features
            text_feat = self.extract_text_features(text)
            text_features.append(text_feat)
        
        # Convert to numpy arrays
        vision_features = np.array(vision_features)
        text_features = np.array(text_features)
        labels = np.array(labels)
        
        result = {
            "vision_features": vision_features,
            "text_features": text_features,
            "labels": labels
        }
        
        if severities is not None:
            result["severities"] = np.array(severities)
        
        return result

class FusionDataPreparer:
    """Prepare fusion training data."""
    
    def __init__(self, config_path: str = "configs/fusion_model.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_extractor = FeatureExtractor(config_path)
        
        # Setup paths
        self.data_config = self.config["data"]
        self.output_dir = Path("data/fusion_features")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_coco_data(self, coco_file: str) -> Dict[str, Any]:
        """Load COCO format data."""
        with open(coco_file, 'r') as f:
            return json.load(f)
    
    def create_fusion_dataset(self, coco_file: str, images_dir: str) -> Dict[str, List]:
        """Create fusion dataset from COCO annotations."""
        
        # Load COCO data
        coco_data = self.load_coco_data(coco_file)
        
        # Create mappings
        image_id_to_info = {img["id"]: img for img in coco_data["images"]}
        
        # Group annotations by image
        image_annotations = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Prepare data
        image_paths = []
        texts = []
        labels = []
        severities = []
        
        for img_id, img_info in image_id_to_info.items():
            if img_id not in image_annotations:
                continue
            
            # Image path
            image_path = Path(images_dir) / img_info["file_name"]
            if not image_path.exists():
                continue
            
            # Get annotations for this image
            annotations = image_annotations[img_id]
            
            # Create text description from annotations
            text_parts = []
            for ann in annotations:
                category_id = ann["category_id"]
                severity = ann.get("severity", 0)
                
                # Get category name
                category_name = next(
                    cat["name"] for cat in coco_data["categories"] 
                    if cat["id"] == category_id
                )
                
                text_parts.append(f"{category_name} with severity {severity}")
            
            # Combine text parts
            text = ". ".join(text_parts) if text_parts else "No specific symptoms observed"
            
            # Use the most severe annotation for this image
            max_severity = max(ann.get("severity", 0) for ann in annotations)
            primary_category = max(annotations, key=lambda x: x.get("severity", 0))["category_id"]
            
            image_paths.append(str(image_path))
            texts.append(text)
            labels.append(primary_category - 1)  # Convert to 0-based indexing
            severities.append(max_severity / 3.0)  # Normalize to 0-1
        
        return {
            "image_paths": image_paths,
            "texts": texts,
            "labels": labels,
            "severities": severities
        }
    
    def prepare_split(self, split_name: str) -> Dict[str, np.ndarray]:
        """Prepare features for a specific split."""
        logger.info(f"Preparing {split_name} split")
        
        # Get paths
        coco_file = self.data_config["fusion"][f"{split_name}_path"]
        images_dir = "data/raw/organized/images"
        
        # Create dataset
        dataset = self.create_fusion_dataset(coco_file, images_dir)
        
        # Extract features
        features = self.feature_extractor.extract_features_batch(
            dataset["image_paths"],
            dataset["texts"],
            dataset["labels"],
            dataset["severities"]
        )
        
        # Save features
        output_file = self.output_dir / f"{split_name}_features.npz"
        np.savez(output_file, **features)
        
        logger.info(f"Saved {split_name} features to {output_file}")
        logger.info(f"Features shape: vision={features['vision_features'].shape}, text={features['text_features'].shape}")
        
        return features
    
    def prepare_all_splits(self):
        """Prepare features for all splits."""
        splits = ["train", "val", "test"]
        
        for split in splits:
            try:
                self.prepare_split(split)
            except Exception as e:
                logger.error(f"Failed to prepare {split} split: {e}")
                continue
        
        logger.info("Feature preparation completed")
    
    def create_metadata(self):
        """Create metadata file for fusion training."""
        metadata = {
            "vision_feature_dim": self.config["model"]["vision"]["feature_dim"],
            "text_feature_dim": self.config["model"]["text"]["feature_dim"],
            "num_classes": self.config["model"]["heads"]["classification"]["num_classes"],
            "feature_extraction_date": str(Path().cwd()),
            "config": self.config
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
    def validate_features(self):
        """Validate extracted features."""
        splits = ["train", "val", "test"]
        
        for split in splits:
            feature_file = self.output_dir / f"{split}_features.npz"
            
            if not feature_file.exists():
                logger.warning(f"Feature file not found: {feature_file}")
                continue
            
            # Load features
            features = np.load(feature_file)
            
            # Validate shapes
            vision_shape = features["vision_features"].shape
            text_shape = features["text_features"].shape
            labels_shape = features["labels"].shape
            
            logger.info(f"{split} features validation:")
            logger.info(f"  Vision features: {vision_shape}")
            logger.info(f"  Text features: {text_shape}")
            logger.info(f"  Labels: {labels_shape}")
            
            # Check for NaN or infinite values
            vision_has_nan = np.isnan(features["vision_features"]).any()
            text_has_nan = np.isnan(features["text_features"]).any()
            
            if vision_has_nan or text_has_nan:
                logger.warning(f"{split} features contain NaN values")
            
            # Check label distribution
            unique_labels, counts = np.unique(features["labels"], return_counts=True)
            logger.info(f"  Label distribution: {dict(zip(unique_labels, counts))}")

def main():
    """Main function to prepare fusion data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare fusion data for AgriSprayAI")
    parser.add_argument("--config", type=str, default="configs/fusion_model.yaml",
                       help="Path to configuration file")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"],
                       default="all", help="Which split to prepare")
    
    args = parser.parse_args()
    
    preparer = FusionDataPreparer(args.config)
    
    if args.split == "all":
        preparer.prepare_all_splits()
    else:
        preparer.prepare_split(args.split)
    
    preparer.create_metadata()
    preparer.validate_features()

if __name__ == "__main__":
    main()
