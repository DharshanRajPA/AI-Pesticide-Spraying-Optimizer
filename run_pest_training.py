#!/usr/bin/env python3
"""
Complete pipeline to process the local agricultural pests dataset and train AgriSprayAI models.
This script handles the entire workflow from dataset processing to model training.
"""

import os
import sys
import logging
from pathlib import Path
import subprocess
import argparse
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgriSprayPipeline:
    """Complete pipeline for processing local pest dataset and training models."""
    
    def __init__(self, dataset_dir: str = "dataset", output_dir: str = "data"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized pipeline with dataset: {self.dataset_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def check_prerequisites(self):
        """Check if all prerequisites are met."""
        logger.info("Checking prerequisites...")
        
        # Check if dataset exists
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        # Check if dataset has expected structure
        expected_categories = [
            'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
            'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
        ]
        
        found_categories = [d.name for d in self.dataset_dir.iterdir() if d.is_dir()]
        missing_categories = set(expected_categories) - set(found_categories)
        
        if missing_categories:
            logger.warning(f"Missing categories: {missing_categories}")
        
        logger.info(f"Found {len(found_categories)} categories: {found_categories}")
        
        # Check Python packages
        required_packages = [
            'torch', 'ultralytics', 'opencv-python', 'PIL', 'numpy', 
            'pandas', 'scikit-learn', 'matplotlib', 'seaborn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            logger.info("Run: pip install -r requirements.txt")
        
        logger.info("Prerequisites check completed")
    
    def process_dataset(self):
        """Process the local dataset into COCO format."""
        logger.info("Step 1: Processing local dataset...")
        
        try:
            # Import and run the dataset processor
            from code.data_prep.process_local_dataset import LocalDatasetProcessor
            
            processor = LocalDatasetProcessor(
                dataset_dir=str(self.dataset_dir),
                output_dir=str(self.output_dir / "annotated")
            )
            
            processor.run()
            
            logger.info("Dataset processing completed successfully")
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            raise
    
    def train_vision_model(self):
        """Train the YOLOv8 pest detection model."""
        logger.info("Step 2: Training vision model...")
        
        try:
            # Import and run the pest detector trainer
            from code.vision.train_pest_detector import PestDetectorTrainer
            
            trainer = PestDetectorTrainer("configs/yolov8_baseline.yaml")
            trainer.run()
            
            logger.info("Vision model training completed successfully")
            
        except Exception as e:
            logger.error(f"Vision model training failed: {e}")
            raise
    
    def prepare_fusion_data(self):
        """Prepare data for multimodal fusion training."""
        logger.info("Step 3: Preparing fusion data...")
        
        try:
            # Import and run the fusion data preparer
            from code.fusion.prepare_fusion_data import FusionDataPreparer
            
            preparer = FusionDataPreparer("configs/fusion_model.yaml")
            preparer.prepare_all_splits()
            preparer.create_metadata()
            preparer.validate_features()
            
            logger.info("Fusion data preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Fusion data preparation failed: {e}")
            raise
    
    def train_fusion_model(self):
        """Train the multimodal fusion model."""
        logger.info("Step 4: Training fusion model...")
        
        try:
            # Import and run the fusion trainer
            from code.fusion.multimodal_fusion import FusionTrainer
            
            trainer = FusionTrainer("configs/fusion_model.yaml")
            trainer.train()
            trainer.evaluate()
            
            logger.info("Fusion model training completed successfully")
            
        except Exception as e:
            logger.error(f"Fusion model training failed: {e}")
            raise
    
    def test_api_server(self):
        """Test the API server with trained models."""
        logger.info("Step 5: Testing API server...")
        
        try:
            # Check if models exist
            model_paths = [
                "models/pest_detector/weights/best.pt",
                "models/fusion_best.pt"
            ]
            
            for model_path in model_paths:
                if not Path(model_path).exists():
                    logger.warning(f"Model not found: {model_path}")
            
            # Start API server in background for testing
            logger.info("API server testing would be performed here")
            logger.info("Run: python code/api/server.py to start the server")
            
        except Exception as e:
            logger.error(f"API server testing failed: {e}")
            raise
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        logger.info("Creating summary report...")
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # Collect statistics
        stats = {
            "pipeline_info": {
                "start_time": self.start_time,
                "end_time": end_time,
                "total_time_minutes": total_time / 60,
                "dataset_dir": str(self.dataset_dir),
                "output_dir": str(self.output_dir)
            },
            "dataset_stats": self._get_dataset_stats(),
            "model_stats": self._get_model_stats(),
            "performance_metrics": self._get_performance_metrics()
        }
        
        # Save report
        import json
        report_file = self.output_dir / "pipeline_summary.json"
        with open(report_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Summary report saved to: {report_file}")
        
        # Print summary
        self._print_summary(stats)
    
    def _get_dataset_stats(self):
        """Get dataset statistics."""
        try:
            stats_file = self.output_dir / "annotated" / "dataset_statistics.json"
            if stats_file.exists():
                import json
                with open(stats_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def _get_model_stats(self):
        """Get model statistics."""
        stats = {}
        
        # Check for trained models
        model_files = [
            "models/pest_detector/weights/best.pt",
            "models/fusion_best.pt",
            "models/segmentation_best.pt"
        ]
        
        for model_file in model_files:
            if Path(model_file).exists():
                stats[model_file] = {
                    "exists": True,
                    "size_mb": Path(model_file).stat().st_size / (1024 * 1024)
                }
            else:
                stats[model_file] = {"exists": False}
        
        return stats
    
    def _get_performance_metrics(self):
        """Get performance metrics from training."""
        metrics = {}
        
        # Check for evaluation results
        eval_files = [
            "models/pest_detector/classification_report.json",
            "models/pest_detector/severity_analysis.json"
        ]
        
        for eval_file in eval_files:
            if Path(eval_file).exists():
                try:
                    import json
                    with open(eval_file, 'r') as f:
                        metrics[eval_file] = json.load(f)
                except:
                    pass
        
        return metrics
    
    def _print_summary(self, stats):
        """Print a comprehensive summary."""
        print("\n" + "="*80)
        print("AGRISPRAYAI PIPELINE EXECUTION SUMMARY")
        print("="*80)
        
        # Pipeline info
        pipeline_info = stats["pipeline_info"]
        print(f"Total Execution Time: {pipeline_info['total_time_minutes']:.1f} minutes")
        print(f"Dataset Directory: {pipeline_info['dataset_dir']}")
        print(f"Output Directory: {pipeline_info['output_dir']}")
        
        # Dataset stats
        if "dataset_stats" in stats and stats["dataset_stats"]:
            dataset_info = stats["dataset_stats"].get("dataset_info", {})
            print(f"\nDataset Statistics:")
            print(f"  Total Categories: {dataset_info.get('total_categories', 'N/A')}")
            print(f"  Total Images: {dataset_info.get('total_images', 'N/A')}")
            print(f"  Average per Category: {dataset_info.get('average_images_per_category', 'N/A'):.1f}")
        
        # Model stats
        model_stats = stats["model_stats"]
        print(f"\nModel Status:")
        for model_path, model_info in model_stats.items():
            status = "✓" if model_info["exists"] else "✗"
            size = f" ({model_info.get('size_mb', 0):.1f} MB)" if model_info["exists"] else ""
            print(f"  {status} {Path(model_path).name}{size}")
        
        # Performance metrics
        if "performance_metrics" in stats and stats["performance_metrics"]:
            print(f"\nPerformance Metrics:")
            for metric_file, metrics in stats["performance_metrics"].items():
                if "accuracy" in metrics:
                    print(f"  Overall Accuracy: {metrics['accuracy']:.3f}")
                if "f1-score" in metrics:
                    print(f"  F1-Score: {metrics['f1-score']:.3f}")
        
        print("\nNext Steps:")
        print("1. Test the API server: python code/api/server.py")
        print("2. Start the React UI: cd ui && npm start")
        print("3. Upload test images through the web interface")
        print("4. Review model performance and fine-tune if needed")
        
        print("="*80)
    
    def run(self, steps: list = None):
        """Run the complete pipeline."""
        if steps is None:
            steps = ["prerequisites", "dataset", "vision", "fusion_data", "fusion", "api_test", "summary"]
        
        logger.info(f"Starting AgriSprayAI pipeline with steps: {steps}")
        
        try:
            if "prerequisites" in steps:
                self.check_prerequisites()
            
            if "dataset" in steps:
                self.process_dataset()
            
            if "vision" in steps:
                self.train_vision_model()
            
            if "fusion_data" in steps:
                self.prepare_fusion_data()
            
            if "fusion" in steps:
                self.train_fusion_model()
            
            if "api_test" in steps:
                self.test_api_server()
            
            if "summary" in steps:
                self.create_summary_report()
            
            logger.info("AgriSprayAI pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run AgriSprayAI training pipeline")
    parser.add_argument("--dataset-dir", type=str, default="dataset",
                       help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Path to output directory")
    parser.add_argument("--steps", nargs="+", 
                       choices=["prerequisites", "dataset", "vision", "fusion_data", "fusion", "api_test", "summary"],
                       default=["prerequisites", "dataset", "vision", "fusion_data", "fusion", "api_test", "summary"],
                       help="Pipeline steps to run")
    parser.add_argument("--skip-dataset", action="store_true",
                       help="Skip dataset processing (use existing processed data)")
    parser.add_argument("--skip-vision", action="store_true",
                       help="Skip vision model training")
    parser.add_argument("--skip-fusion", action="store_true",
                       help="Skip fusion model training")
    
    args = parser.parse_args()
    
    # Modify steps based on skip flags
    steps = args.steps.copy()
    if args.skip_dataset:
        steps = [s for s in steps if s not in ["dataset", "fusion_data"]]
    if args.skip_vision:
        steps = [s for s in steps if s != "vision"]
    if args.skip_fusion:
        steps = [s for s in steps if s not in ["fusion_data", "fusion"]]
    
    # Initialize and run pipeline
    pipeline = AgriSprayPipeline(args.dataset_dir, args.output_dir)
    pipeline.run(steps)

if __name__ == "__main__":
    main()
