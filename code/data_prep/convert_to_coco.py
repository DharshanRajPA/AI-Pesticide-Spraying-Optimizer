#!/usr/bin/env python3
"""
Convert agricultural pest dataset to COCO format with custom severity field.
This script handles the conversion from various formats to COCO format
required for AgriSprayAI training.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class COCOConverter:
    """Converts agricultural pest dataset to COCO format with severity annotations."""
    
    def __init__(self, 
                 input_dir: str = "data/raw/organized",
                 output_dir: str = "data/annotated"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO format structure
        self.coco_format = {
            "info": {
                "description": "Agricultural Pests Dataset for AgriSprayAI",
                "version": "1.0",
                "year": 2023,
                "contributor": "AgriSprayAI Team",
                "date_created": "2023-01-01"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Define categories with severity mapping
        self.categories = [
            {"id": 1, "name": "aphid", "supercategory": "pest"},
            {"id": 2, "name": "armyworm", "supercategory": "pest"},
            {"id": 3, "name": "beetle", "supercategory": "pest"},
            {"id": 4, "name": "bollworm", "supercategory": "pest"},
            {"id": 5, "name": "grasshopper", "supercategory": "pest"},
            {"id": 6, "name": "mites", "supercategory": "pest"},
            {"id": 7, "name": "mosquito", "supercategory": "pest"},
            {"id": 8, "name": "sawfly", "supercategory": "pest"},
            {"id": 9, "name": "stem_borer", "supercategory": "pest"},
            {"id": 10, "name": "healthy", "supercategory": "healthy"}
        ]
        
        # Severity mapping (can be customized based on expert knowledge)
        self.severity_mapping = {
            "aphid": {"mild": 1, "moderate": 2, "severe": 3},
            "armyworm": {"mild": 1, "moderate": 2, "severe": 3},
            "beetle": {"mild": 1, "moderate": 2, "severe": 3},
            "bollworm": {"mild": 1, "moderate": 2, "severe": 3},
            "grasshopper": {"mild": 1, "moderate": 2, "severe": 3},
            "mites": {"mild": 1, "moderate": 2, "severe": 3},
            "mosquito": {"mild": 1, "moderate": 2, "severe": 3},
            "sawfly": {"mild": 1, "moderate": 2, "severe": 3},
            "stem_borer": {"mild": 1, "moderate": 2, "severe": 3},
            "healthy": {"none": 0}
        }
    
    def load_existing_annotations(self) -> Dict[str, Any]:
        """Load existing annotations if available."""
        annotations_file = self.input_dir / "annotations" / "instances_all.json"
        
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning("No existing annotations found, creating placeholder annotations")
            return None
    
    def process_images(self) -> List[Dict[str, Any]]:
        """Process all images and extract metadata."""
        images_dir = self.input_dir / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        logger.info(f"Found {len(image_files)} images to process")
        
        images = []
        for i, img_path in enumerate(image_files):
            try:
                # Load image to get dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                
                image_info = {
                    "id": i + 1,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                    "date_captured": "2023-01-01",  # Placeholder
                    "license": 1,
                    "coco_url": "",
                    "flickr_url": "",
                    "farm_id": f"farm_{i % 10}",  # Simulate farm IDs
                    "capture_device": "mobile" if i % 2 == 0 else "uav",
                    "gps": {
                        "latitude": 40.7128 + (i * 0.001),
                        "longitude": -74.0060 + (i * 0.001)
                    }
                }
                images.append(image_info)
                
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
                continue
        
        return images
    
    def create_annotations(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create annotations for the images."""
        annotations = []
        annotation_id = 1
        
        for img_info in images:
            # For demonstration, create 1-3 annotations per image
            num_annotations = np.random.randint(1, 4)
            
            for _ in range(num_annotations):
                # Random category (excluding healthy for now)
                category_id = np.random.randint(1, 10)
                category_name = next(cat["name"] for cat in self.categories if cat["id"] == category_id)
                
                # Random severity based on category
                if category_name == "healthy":
                    severity = 0
                else:
                    severity = np.random.randint(1, 4)
                
                # Random bounding box (10-50% of image)
                img_width, img_height = img_info["width"], img_info["height"]
                bbox_width = np.random.randint(int(img_width * 0.1), int(img_width * 0.5))
                bbox_height = np.random.randint(int(img_height * 0.1), int(img_height * 0.5))
                bbox_x = np.random.randint(0, img_width - bbox_width)
                bbox_y = np.random.randint(0, img_height - bbox_height)
                
                # Create segmentation polygon (simplified rectangle)
                segmentation = [
                    [bbox_x, bbox_y, 
                     bbox_x + bbox_width, bbox_y,
                     bbox_x + bbox_width, bbox_y + bbox_height,
                     bbox_x, bbox_y + bbox_height]
                ]
                
                annotation = {
                    "id": annotation_id,
                    "image_id": img_info["id"],
                    "category_id": category_id,
                    "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                    "severity": severity  # Custom field for AgriSprayAI
                }
                
                annotations.append(annotation)
                annotation_id += 1
        
        return annotations
    
    def split_dataset(self, images: List[Dict[str, Any]], 
                     annotations: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
        """Split dataset into train/val/test sets with farm-wise splitting."""
        
        # Group images by farm_id for proper splitting
        farm_groups = {}
        for img in images:
            farm_id = img["farm_id"]
            if farm_id not in farm_groups:
                farm_groups[farm_id] = []
            farm_groups[farm_id].append(img)
        
        # Split farms into train/val/test
        farm_ids = list(farm_groups.keys())
        train_farms, temp_farms = train_test_split(farm_ids, test_size=0.3, random_state=42)
        val_farms, test_farms = train_test_split(temp_farms, test_size=0.5, random_state=42)
        
        # Create splits
        splits = {"train": train_farms, "val": val_farms, "test": test_farms}
        
        result_splits = {}
        for split_name, farm_list in splits.items():
            # Get images for this split
            split_images = []
            for farm_id in farm_list:
                split_images.extend(farm_groups[farm_id])
            
            # Get annotations for this split
            split_image_ids = {img["id"] for img in split_images}
            split_annotations = [ann for ann in annotations if ann["image_id"] in split_image_ids]
            
            # Create COCO format for this split
            coco_split = {
                "info": self.coco_format["info"].copy(),
                "licenses": self.coco_format["licenses"].copy(),
                "images": split_images,
                "annotations": split_annotations,
                "categories": self.categories.copy()
            }
            
            result_splits[split_name] = coco_split
            
            logger.info(f"{split_name}: {len(split_images)} images, {len(split_annotations)} annotations")
        
        return result_splits["train"], result_splits["val"], result_splits["test"]
    
    def save_splits(self, train_data: Dict[str, Any], 
                   val_data: Dict[str, Any], 
                   test_data: Dict[str, Any]) -> None:
        """Save the dataset splits to files."""
        
        # Save train split
        train_file = self.output_dir / "instances_train.json"
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        logger.info(f"Saved train split: {train_file}")
        
        # Save validation split
        val_file = self.output_dir / "instances_val.json"
        with open(val_file, 'w') as f:
            json.dump(val_data, f, indent=2)
        logger.info(f"Saved validation split: {val_file}")
        
        # Save test split
        test_file = self.output_dir / "instances_test.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        logger.info(f"Saved test split: {test_file}")
        
        # Save dataset statistics
        stats = {
            "total_images": len(train_data["images"]) + len(val_data["images"]) + len(test_data["images"]),
            "total_annotations": len(train_data["annotations"]) + len(val_data["annotations"]) + len(test_data["annotations"]),
            "train": {
                "images": len(train_data["images"]),
                "annotations": len(train_data["annotations"])
            },
            "val": {
                "images": len(val_data["images"]),
                "annotations": len(val_data["annotations"])
            },
            "test": {
                "images": len(test_data["images"]),
                "annotations": len(test_data["annotations"])
            },
            "categories": len(self.categories),
            "severity_distribution": self._calculate_severity_distribution(train_data["annotations"])
        }
        
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved dataset statistics: {stats_file}")
    
    def _calculate_severity_distribution(self, annotations: List[Dict[str, Any]]) -> Dict[int, int]:
        """Calculate severity distribution in annotations."""
        severity_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for ann in annotations:
            severity = ann.get("severity", 0)
            severity_counts[severity] += 1
        return severity_counts
    
    def create_masks_directory(self) -> None:
        """Create directory structure for segmentation masks."""
        masks_dir = Path("data/masks")
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each split
        for split in ["train", "val", "test"]:
            (masks_dir / split).mkdir(exist_ok=True)
        
        logger.info("Created masks directory structure")
    
    def run(self) -> None:
        """Run the complete conversion process."""
        try:
            logger.info("Starting COCO conversion process")
            
            # Process images
            images = self.process_images()
            logger.info(f"Processed {len(images)} images")
            
            # Create annotations
            annotations = self.create_annotations(images)
            logger.info(f"Created {len(annotations)} annotations")
            
            # Split dataset
            train_data, val_data, test_data = self.split_dataset(images, annotations)
            
            # Save splits
            self.save_splits(train_data, val_data, test_data)
            
            # Create masks directory
            self.create_masks_directory()
            
            logger.info("COCO conversion completed successfully")
            
        except Exception as e:
            logger.error(f"COCO conversion failed: {e}")
            raise

def main():
    """Main function to run the COCO converter."""
    converter = COCOConverter()
    converter.run()

if __name__ == "__main__":
    main()
