#!/usr/bin/env python3
"""
Process the local agricultural pests dataset for AgriSprayAI.
Converts the folder-based structure to COCO format with severity annotations.
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
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalDatasetProcessor:
    """Process the local agricultural pests dataset."""
    
    def __init__(self, dataset_dir: str = "dataset", output_dir: str = "data/annotated"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Category to severity mapping based on agricultural impact
        self.category_severity_mapping = {
            'catterpillar': 3,  # High severity - severe defoliation
            'grasshopper': 3,   # High severity - major crop damage
            'weevil': 3,        # High severity - stored grain damage
            'slug': 3,          # High severity - rapid leaf damage
            'snail': 3,         # High severity - similar to slugs
            'beetle': 2,        # Medium severity - moderate damage
            'moth': 2,          # Medium severity - caterpillar potential
            'earwig': 2,        # Medium severity - minor fruit damage
            'ants': 1,          # Low severity - can protect aphids
            'wasp': 1,          # Low severity - mixed impact
            'bees': 0,          # Beneficial - essential pollinators
            'earthworms': 0     # Beneficial - soil health
        }
        
        # Category descriptions for text processing
        self.category_descriptions = {
            'ants': 'Small social insects that can protect aphids and cause minor root damage',
            'bees': 'Essential pollinators crucial for crop production and ecosystem health',
            'beetle': 'Coleoptera insects that cause moderate leaf damage and can affect crop yield',
            'catterpillar': 'Lepidoptera larvae that cause severe defoliation and major crop damage',
            'earthworms': 'Beneficial soil organisms that improve soil structure and nutrient cycling',
            'earwig': 'Dermaptera insects that can cause minor fruit damage in some crops',
            'grasshopper': 'Orthoptera insects that cause major crop damage through defoliation',
            'moth': 'Adult Lepidoptera that can lay eggs leading to caterpillar infestations',
            'slug': 'Gastropods that cause rapid leaf damage, especially in wet conditions',
            'snail': 'Gastropods similar to slugs that damage leaves and young plants',
            'wasp': 'Hymenoptera insects with mixed impact - some are beneficial predators',
            'weevil': 'Coleoptera beetles that cause major damage to stored grains and crops'
        }
        
        # Statistics
        self.stats = defaultdict(int)
        
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze the dataset structure and return statistics."""
        logger.info("Analyzing dataset structure...")
        
        categories = []
        total_images = 0
        
        for category_dir in self.dataset_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                
                categories.append({
                    'name': category_name,
                    'count': len(image_files),
                    'severity': self.category_severity_mapping.get(category_name, 1),
                    'description': self.category_descriptions.get(category_name, 'Unknown pest')
                })
                
                total_images += len(image_files)
                self.stats[category_name] = len(image_files)
        
        # Sort categories by count
        categories.sort(key=lambda x: x['count'], reverse=True)
        
        analysis = {
            'total_categories': len(categories),
            'total_images': total_images,
            'categories': categories,
            'severity_distribution': self._get_severity_distribution(categories),
            'average_images_per_category': total_images / len(categories)
        }
        
        logger.info(f"Dataset analysis complete: {total_images} images across {len(categories)} categories")
        return analysis
    
    def _get_severity_distribution(self, categories: List[Dict]) -> Dict[int, int]:
        """Get distribution of images by severity level."""
        severity_dist = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for category in categories:
            severity = category['severity']
            severity_dist[severity] += category['count']
        
        return severity_dist
    
    def create_coco_annotations(self, split: str = "all") -> Dict[str, Any]:
        """Create COCO format annotations for the dataset."""
        logger.info(f"Creating COCO annotations for split: {split}")
        
        # Create categories
        categories = []
        for i, (category_name, severity) in enumerate(self.category_severity_mapping.items(), 1):
            categories.append({
                "id": i,
                "name": category_name,
                "supercategory": "pest" if severity > 0 else "beneficial",
                "severity": severity,
                "description": self.category_descriptions.get(category_name, "")
            })
        
        # Create images and annotations
        images = []
        annotations = []
        image_id = 1
        annotation_id = 1
        
        for category_dir in self.dataset_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            category_id = next(cat["id"] for cat in categories if cat["name"] == category_name)
            severity = self.category_severity_mapping.get(category_name, 1)
            
            # Get image files
            image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
            
            for img_file in image_files:
                try:
                    # Load image to get dimensions
                    with Image.open(img_file) as img:
                        width, height = img.size
                    
                    # Create image entry
                    image_entry = {
                        "id": image_id,
                        "file_name": f"{category_name}/{img_file.name}",
                        "width": width,
                        "height": height,
                        "date_captured": "2023-01-01",
                        "license": 1,
                        "coco_url": "",
                        "flickr_url": "",
                        "category": category_name,
                        "severity": severity
                    }
                    images.append(image_entry)
                    
                    # Create annotation (simplified bounding box covering most of the image)
                    # In a real scenario, you'd have proper bounding box annotations
                    bbox_width = int(width * 0.8)  # 80% of image width
                    bbox_height = int(height * 0.8)  # 80% of image height
                    bbox_x = int(width * 0.1)  # 10% margin
                    bbox_y = int(height * 0.1)  # 10% margin
                    
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "segmentation": [
                            [bbox_x, bbox_y, 
                             bbox_x + bbox_width, bbox_y,
                             bbox_x + bbox_width, bbox_y + bbox_height,
                             bbox_x, bbox_y + bbox_height]
                        ],
                        "iscrowd": 0,
                        "severity": severity,
                        "confidence": 1.0  # Ground truth
                    }
                    annotations.append(annotation)
                    
                    image_id += 1
                    annotation_id += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
                    continue
        
        # Create COCO format
        coco_data = {
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
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        logger.info(f"Created COCO annotations: {len(images)} images, {len(annotations)} annotations")
        return coco_data
    
    def create_train_val_test_split(self, coco_data: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
        """Create train/validation/test splits maintaining category balance."""
        logger.info("Creating train/validation/test splits...")
        
        # Group images by category
        category_images = defaultdict(list)
        for img in coco_data["images"]:
            category = img["category"]
            category_images[category].append(img)
        
        # Create splits for each category
        train_images = []
        val_images = []
        test_images = []
        
        train_annotations = []
        val_annotations = []
        test_annotations = []
        
        for category, images in category_images.items():
            # Split images: 70% train, 15% val, 15% test
            train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
            
            train_images.extend(train_imgs)
            val_images.extend(val_imgs)
            test_images.extend(test_imgs)
        
        # Create annotation mappings
        train_image_ids = {img["id"] for img in train_images}
        val_image_ids = {img["id"] for img in val_images}
        test_image_ids = {img["id"] for img in test_images}
        
        # Filter annotations
        for ann in coco_data["annotations"]:
            if ann["image_id"] in train_image_ids:
                train_annotations.append(ann)
            elif ann["image_id"] in val_image_ids:
                val_annotations.append(ann)
            elif ann["image_id"] in test_image_ids:
                test_annotations.append(ann)
        
        # Create split datasets
        train_data = {
            "info": coco_data["info"],
            "licenses": coco_data["licenses"],
            "images": train_images,
            "annotations": train_annotations,
            "categories": coco_data["categories"]
        }
        
        val_data = {
            "info": coco_data["info"],
            "licenses": coco_data["licenses"],
            "images": val_images,
            "annotations": val_annotations,
            "categories": coco_data["categories"]
        }
        
        test_data = {
            "info": coco_data["info"],
            "licenses": coco_data["licenses"],
            "images": test_images,
            "annotations": test_annotations,
            "categories": coco_data["categories"]
        }
        
        logger.info(f"Split created - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
        return train_data, val_data, test_data
    
    def copy_images_to_organized_structure(self):
        """Copy images to organized structure for training."""
        logger.info("Organizing images for training...")
        
        organized_dir = Path("data/raw/organized")
        images_dir = organized_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        for category_dir in self.dataset_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
            
            for img_file in image_files:
                try:
                    # Create new filename with category prefix
                    new_filename = f"{category_name}_{img_file.name}"
                    dest_path = images_dir / new_filename
                    
                    # Copy image
                    shutil.copy2(img_file, dest_path)
                    copied_count += 1
                    
                except Exception as e:
                    logger.error(f"Error copying {img_file}: {e}")
                    continue
        
        logger.info(f"Copied {copied_count} images to organized structure")
    
    def save_annotations(self, train_data: Dict, val_data: Dict, test_data: Dict):
        """Save the split annotations to files."""
        logger.info("Saving annotations...")
        
        # Save train split
        train_file = self.output_dir / "instances_train.json"
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        # Save validation split
        val_file = self.output_dir / "instances_val.json"
        with open(val_file, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        # Save test split
        test_file = self.output_dir / "instances_test.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"Saved annotations to {self.output_dir}")
    
    def create_dataset_statistics(self, analysis: Dict[str, Any]):
        """Create comprehensive dataset statistics."""
        stats_file = self.output_dir / "dataset_statistics.json"
        
        statistics = {
            "dataset_info": analysis,
            "category_mapping": self.category_severity_mapping,
            "category_descriptions": self.category_descriptions,
            "processing_info": {
                "total_processed": sum(self.stats.values()),
                "categories_processed": len(self.stats),
                "severity_distribution": analysis["severity_distribution"]
            }
        }
        
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        logger.info(f"Saved dataset statistics to {stats_file}")
    
    def run(self):
        """Run the complete dataset processing pipeline."""
        try:
            logger.info("Starting local dataset processing...")
            
            # Step 1: Analyze dataset
            analysis = self.analyze_dataset()
            
            # Step 2: Create COCO annotations
            coco_data = self.create_coco_annotations()
            
            # Step 3: Create train/val/test splits
            train_data, val_data, test_data = self.create_train_val_test_split(coco_data)
            
            # Step 4: Copy images to organized structure
            self.copy_images_to_organized_structure()
            
            # Step 5: Save annotations
            self.save_annotations(train_data, val_data, test_data)
            
            # Step 6: Create statistics
            self.create_dataset_statistics(analysis)
            
            logger.info("Dataset processing completed successfully!")
            
            # Print summary
            print("\n" + "="*50)
            print("DATASET PROCESSING SUMMARY")
            print("="*50)
            print(f"Total Categories: {analysis['total_categories']}")
            print(f"Total Images: {analysis['total_images']}")
            print(f"Average per Category: {analysis['average_images_per_category']:.1f}")
            print("\nSeverity Distribution:")
            for severity, count in analysis['severity_distribution'].items():
                severity_name = {0: 'Beneficial', 1: 'Low', 2: 'Medium', 3: 'High'}[severity]
                print(f"  {severity_name}: {count} images")
            print("\nCategory Breakdown:")
            for category in analysis['categories']:
                print(f"  {category['name']}: {category['count']} images (Severity: {category['severity']})")
            print("="*50)
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            raise

def main():
    """Main function to run the dataset processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process local agricultural pests dataset")
    parser.add_argument("--dataset-dir", type=str, default="dataset",
                       help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="data/annotated",
                       help="Path to output directory")
    
    args = parser.parse_args()
    
    processor = LocalDatasetProcessor(args.dataset_dir, args.output_dir)
    processor.run()

if __name__ == "__main__":
    main()
