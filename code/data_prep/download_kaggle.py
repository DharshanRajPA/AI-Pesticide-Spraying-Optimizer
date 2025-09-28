#!/usr/bin/env python3
"""
Download and prepare the Kaggle Agricultural Pests Image Dataset
for AgriSprayAI training and evaluation.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleDatasetDownloader:
    """Downloads and prepares the Kaggle Agricultural Pests dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kaggle API
        self.api = KaggleApi()
        self.api.authenticate()
        
        # Dataset information
        self.dataset_name = "vencerlanz09/agricultural-pests-image-dataset"
        self.dataset_dir = self.data_dir / "kaggle_pests"
        
    def download_dataset(self) -> None:
        """Download the dataset from Kaggle."""
        logger.info(f"Downloading dataset: {self.dataset_name}")
        
        try:
            # Download dataset files
            self.api.dataset_download_files(
                self.dataset_name,
                path=str(self.data_dir),
                unzip=True
            )
            logger.info("Dataset downloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def organize_dataset(self) -> None:
        """Organize the downloaded dataset into a structured format."""
        logger.info("Organizing dataset structure")
        
        # Create organized directory structure
        organized_dir = self.data_dir / "organized"
        organized_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (organized_dir / "images").mkdir(exist_ok=True)
        (organized_dir / "annotations").mkdir(exist_ok=True)
        (organized_dir / "metadata").mkdir(exist_ok=True)
        
        # Find and organize files
        kaggle_dir = self.data_dir / "agricultural-pests-image-dataset"
        if not kaggle_dir.exists():
            logger.error("Kaggle dataset directory not found")
            return
        
        # Copy images
        image_files = list(kaggle_dir.glob("**/*.jpg")) + list(kaggle_dir.glob("**/*.png"))
        for img_file in image_files:
            dest_path = organized_dir / "images" / img_file.name
            shutil.copy2(img_file, dest_path)
        
        logger.info(f"Organized {len(image_files)} images")
        
        # Create basic metadata
        self._create_basic_metadata(organized_dir, image_files)
    
    def _create_basic_metadata(self, organized_dir: Path, image_files: List[Path]) -> None:
        """Create basic metadata for the dataset."""
        metadata = {
            "dataset_name": "Agricultural Pests Image Dataset",
            "source": "Kaggle",
            "total_images": len(image_files),
            "categories": [
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
            ],
            "image_info": []
        }
        
        # Add image information
        for i, img_file in enumerate(image_files):
            img_info = {
                "id": i + 1,
                "file_name": img_file.name,
                "width": 0,  # Will be filled during processing
                "height": 0,  # Will be filled during processing
                "date_captured": "2023-01-01",  # Placeholder
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            }
            metadata["image_info"].append(img_info)
        
        # Save metadata
        metadata_file = organized_dir / "metadata" / "dataset_info.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created metadata for {len(image_files)} images")
    
    def create_placeholder_annotations(self) -> None:
        """Create placeholder annotations for the dataset."""
        logger.info("Creating placeholder annotations")
        
        organized_dir = self.data_dir / "organized"
        metadata_file = organized_dir / "metadata" / "dataset_info.json"
        
        if not metadata_file.exists():
            logger.error("Metadata file not found")
            return
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create COCO format annotations
        coco_annotations = {
            "info": {
                "description": "Agricultural Pests Dataset",
                "version": "1.0",
                "year": 2023,
                "contributor": "AgriSprayAI",
                "date_created": "2023-01-01"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": metadata["image_info"],
            "annotations": [],
            "categories": metadata["categories"]
        }
        
        # Create placeholder annotations (will be replaced with actual annotations)
        annotation_id = 1
        for img_info in metadata["image_info"]:
            # Create a placeholder annotation for each image
            # In practice, these would be created by annotation tools
            annotation = {
                "id": annotation_id,
                "image_id": img_info["id"],
                "category_id": 1,  # Placeholder category
                "bbox": [0, 0, 100, 100],  # Placeholder bbox
                "area": 10000,
                "segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]],  # Placeholder segmentation
                "iscrowd": 0,
                "severity": 1  # Custom field for severity (0-3)
            }
            coco_annotations["annotations"].append(annotation)
            annotation_id += 1
        
        # Save annotations
        annotations_file = organized_dir / "annotations" / "instances_all.json"
        with open(annotations_file, 'w') as f:
            json.dump(coco_annotations, f, indent=2)
        
        logger.info(f"Created {len(coco_annotations['annotations'])} placeholder annotations")
    
    def run(self) -> None:
        """Run the complete download and preparation process."""
        try:
            self.download_dataset()
            self.organize_dataset()
            self.create_placeholder_annotations()
            logger.info("Dataset preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            raise

def main():
    """Main function to run the dataset downloader."""
    downloader = KaggleDatasetDownloader()
    downloader.run()

if __name__ == "__main__":
    main()
