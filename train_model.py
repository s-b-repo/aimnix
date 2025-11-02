#!/usr/bin/env python3
"""
CS2 Head Detection Model Training Script
Enhanced version with team detection and advanced features
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Check if ultralytics is installed
try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
except ImportError as e:
    print(f"[!] Required packages not installed: {e}")
    print("[!] Run: pip install ultralytics opencv-python numpy")
    sys.exit(1)

class CS2ModelTrainer:
    def __init__(self):
        self.project_name = "cs2_head_detection"
        self.model_size = "n"  # nano model for speed
        self.dataset_path = "cs2_dataset"
        self.config = {
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "confidence": 0.35,
            "iou": 0.7,
            "augment": True,
            "device": "0"  # GPU device ID
        }
    
    def create_dataset_structure(self):
        """Create the required dataset directory structure"""
        print("[+] Creating dataset structure...")
        
        directories = [
            "cs2_dataset/images/train",
            "cs2_dataset/images/val", 
            "cs2_dataset/images/test",
            "cs2_dataset/labels/train",
            "cs2_dataset/labels/val",
            "cs2_dataset/labels/test"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        # Create data.yaml
        data_yaml = f"""
path: {os.path.abspath(self.dataset_path)}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['head']
"""
        
        with open(f"{self.dataset_path}/data.yaml", "w") as f:
            f.write(data_yaml.strip())
            
        print(f"[✓] Dataset structure created at {self.dataset_path}")
    
    def check_dataset(self):
        """Verify dataset integrity"""
        print("[+] Checking dataset...")
        
        train_images = list(Path(f"{self.dataset_path}/images/train").glob("*.jpg"))
        val_images = list(Path(f"{self.dataset_path}/images/val").glob("*.jpg"))
        
        if len(train_images) == 0:
            print("[!] No training images found!")
            print("[!] Please add images to cs2_dataset/images/train/")
            return False
            
        if len(val_images) == 0:
            print("[!] No validation images found!")
            print("[!] Please add images to cs2_dataset/images/val/")
            return False
            
        print(f"[✓] Found {len(train_images)} training images")
        print(f"[✓] Found {len(val_images)} validation images")
        return True
    
    def train_model(self):
        """Train the YOLO model"""
        print("[+] Starting model training...")
        print(f"[+] Using YOLOv8{self.model_size} model")
        print(f"[+] Training for {self.config['epochs']} epochs")
        
        try:
            # Load model
            model = YOLO(f"yolov8{self.model_size}.yaml")
            
            # Start training
            results = model.train(
                data=f"{self.dataset_path}/data.yaml",
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch'],
                name=f"cs2_head_{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                device=self.config['device'],
                augment=self.config['augment'],
                verbose=True
            )
            
            print("[✓] Training completed successfully!")
            return model
            
        except Exception as e:
            print(f"[!] Training failed: {e}")
            return None
    
    def export_model(self, model, export_format="onnx"):
        """Export the trained model"""
        print(f"[+] Exporting model to {export_format.upper()}...")
        
        try:
            # Export model
            exported_path = model.export(
                format=export_format,
                imgsz=self.config['imgsz'],
                opset=12,
                optimize=True
            )
            
            # Copy to aimbot directory
            if os.path.exists(exported_path):
                shutil.copy(exported_path, "cs2head.onnx")
                print(f"[✓] Model exported to cs2head.onnx")
                return True
            else:
                print(f"[!] Export failed - file not found")
                return False
                
        except Exception as e:
            print(f"[!] Export failed: {e}")
            return False
    
    def validate_model(self, model_path="cs2head.onnx"):
        """Validate the exported model"""
        print("[+] Validating model...")
        
        try:
            # Load and test the model
            model = YOLO(model_path)
            
            # Test on a sample image if available
            test_images = list(Path("cs2_dataset/images/val").glob("*.jpg"))[:5]
            
            if test_images:
                print("[+] Testing model on validation images...")
                results = model(test_images)
                
                # Print detection statistics
                for i, result in enumerate(results):
                    boxes = result.boxes
                    if boxes is not None:
                        print(f"  Image {i+1}: {len(boxes)} detections")
                
                print("[✓] Model validation completed")
                return True
            else:
                print("[✓] Model loaded successfully")
                return True
                
        except Exception as e:
            print(f"[!] Model validation failed: {e}")
            return False
    
    def create_training_script(self):
        """Create a reusable training script"""
        script_content = '''#!/usr/bin/env python3
"""Quick training script for CS2 head detection"""

from train_model import CS2ModelTrainer

def main():
    trainer = CS2ModelTrainer()
    
    # Check if dataset exists
    if not trainer.check_dataset():
        print("[!] Please prepare your dataset first!")
        print("[!] Add images to cs2_dataset/images/train/ and cs2_dataset/images/val/")
        return
    
    # Train model
    model = trainer.train_model()
    if model:
        # Export model
        if trainer.export_model(model):
            print("[✓] Model training and export completed!")
            print("[+] Your model is ready: cs2head.onnx")
        else:
            print("[!] Model export failed!")
    else:
        print("[!] Model training failed!")

if __name__ == "__main__":
    main()
'''
        
        with open("quick_train.py", "w") as f:
            f.write(script_content)
        
        os.chmod("quick_train.py", 0o755)
        print("[✓] Created quick training script: quick_train.py")

def main():
    parser = argparse.ArgumentParser(description="CS2 Head Detection Model Training")
    parser.add_argument("--setup", action="store_true", help="Set up dataset structure")
    parser.add_argument("--check", action="store_true", help="Check dataset integrity")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--export", action="store_true", help="Export trained model")
    parser.add_argument("--validate", action="store_true", help="Validate model")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--model", choices=["n", "s", "m", "l", "x"], default="n", help="YOLO model size")
    parser.add_argument("--device", default="0", help="GPU device ID or 'cpu'")
    
    args = parser.parse_args()
    
    trainer = CS2ModelTrainer()
    trainer.config['epochs'] = args.epochs
    trainer.model_size = args.model
    trainer.config['device'] = args.device
    
    if args.setup or args.all:
        trainer.create_dataset_structure()
    
    if args.check or args.all:
        if not trainer.check_dataset():
            print("[!] Dataset check failed. Please prepare your data.")
            return
    
    if args.train or args.all:
        model = trainer.train_model()
        if not model:
            print("[!] Training failed!")
            return
    
    if args.export or args.all:
        if 'model' in locals():
            trainer.export_model(model)
        else:
            print("[!] No model to export. Run training first.")
    
    if args.validate or args.all:
        trainer.validate_model()
    
    # Always create the quick training script
    trainer.create_training_script()
    
    if not any([args.setup, args.check, args.train, args.export, args.validate, args.all]):
        print("CS2 Head Detection Model Trainer")
        print("Usage: python3 train_model.py [OPTIONS]")
        print("\nOptions:")
        print("  --setup     Create dataset structure")
        print("  --check     Check dataset integrity")
        print("  --train     Train the model")
        print("  --export    Export trained model")
        print("  --validate  Validate model")
        print("  --all       Run complete pipeline")
        print("  --epochs N  Set number of epochs (default: 100)")
        print("  --model SIZE YOLO model size: n,s,m,l,x (default: n)")
        print("  --device ID GPU device or 'cpu' (default: 0)")

if __name__ == "__main__":
    main()