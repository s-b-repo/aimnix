#!/usr/bin/env python3
"""
CS2 Head Detection Model Training Script
"""

import os
import sys
import argparse
import shutil
import random
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
except ImportError as e:
    print(f"[!] Required packages not installed: {e}")
    print("[!] Run: pip install ultralytics opencv-python numpy")
    sys.exit(1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def find_images(directory):
    path = Path(directory)
    if not path.exists():
        return []
    return [f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]


class CS2ModelTrainer:
    def __init__(self):
        self.dataset_path = "cs2_dataset"
        self.model_size = "n"
        self.config = {
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "lr0": 0.01,
            "lrf": 0.01,
            "device": "0",
            "augment": True,
            "resume": False,
        }
        self._trained_model = None

    def create_dataset_structure(self):
        print("[+] Creating dataset structure...")

        for split in ("train", "val", "test"):
            Path(f"{self.dataset_path}/images/{split}").mkdir(parents=True, exist_ok=True)
            Path(f"{self.dataset_path}/labels/{split}").mkdir(parents=True, exist_ok=True)

        data_yaml = (
            f"path: {os.path.abspath(self.dataset_path)}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "\n"
            "nc: 1\n"
            "names: ['head']\n"
        )
        with open(f"{self.dataset_path}/data.yaml", "w") as f:
            f.write(data_yaml)

        print(f"[+] Dataset structure created at {self.dataset_path}")

    def auto_split(self, val_ratio=0.15, seed=42):
        """Move a fraction of train images/labels to val/ if val/ is empty."""
        train_img_dir = Path(f"{self.dataset_path}/images/train")
        val_img_dir = Path(f"{self.dataset_path}/images/val")
        train_lbl_dir = Path(f"{self.dataset_path}/labels/train")
        val_lbl_dir = Path(f"{self.dataset_path}/labels/val")

        val_images = find_images(val_img_dir)
        if val_images:
            print(f"[+] Val set already has {len(val_images)} images, skipping auto-split")
            return

        train_images = find_images(train_img_dir)
        if not train_images:
            print("[!] No training images to split")
            return

        rng = random.Random(seed)
        rng.shuffle(train_images)
        n_val = max(1, int(len(train_images) * val_ratio))
        to_move = train_images[:n_val]

        moved = 0
        for img_path in to_move:
            try:
                shutil.move(str(img_path), str(val_img_dir / img_path.name))
                moved += 1
                label_path = train_lbl_dir / img_path.with_suffix(".txt").name
                if label_path.exists():
                    shutil.move(str(label_path), str(val_lbl_dir / label_path.name))
            except OSError as e:
                print(f"[!] Failed to move {img_path.name}: {e}")

        print(f"[+] Auto-split: moved {moved} images to val/")

    def check_dataset(self):
        print("[+] Checking dataset...")

        train_images = find_images(f"{self.dataset_path}/images/train")
        val_images = find_images(f"{self.dataset_path}/images/val")

        train_labels = list(Path(f"{self.dataset_path}/labels/train").glob("*.txt"))
        val_labels = list(Path(f"{self.dataset_path}/labels/val").glob("*.txt"))

        if not train_images:
            print("[!] No training images found!")
            print(f"[!] Add images to {self.dataset_path}/images/train/")
            return False

        if not val_images:
            print("[!] No validation images found!")
            print("[!] Run with --split to auto-split from training set")
            return False

        print(f"[+] Train: {len(train_images)} images, {len(train_labels)} labels")
        print(f"[+] Val:   {len(val_images)} images, {len(val_labels)} labels")

        unlabeled = len(train_images) - len(train_labels)
        if unlabeled > 0:
            print(f"[!] Warning: {unlabeled} training images have no labels")

        return True

    def train_model(self):
        print(f"[+] Training YOLOv8{self.model_size} for {self.config['epochs']} epochs...")

        try:
            if self.config["resume"]:
                last_pt = self._find_last_checkpoint()
                if last_pt:
                    print(f"[+] Resuming from checkpoint: {last_pt}")
                    model = YOLO(last_pt)
                    results = model.train(resume=True)
                    self._trained_model = model
                    print("[+] Training resumed and completed!")
                    return model

            model = YOLO(f"yolov8{self.model_size}.yaml")

            run_name = f"cs2_head_{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            results = model.train(
                data=f"{self.dataset_path}/data.yaml",
                epochs=self.config["epochs"],
                imgsz=self.config["imgsz"],
                batch=self.config["batch"],
                lr0=self.config["lr0"],
                lrf=self.config["lrf"],
                name=run_name,
                device=self.config["device"],
                augment=self.config["augment"],
                verbose=True,
            )

            self._trained_model = model
            print("[+] Training completed!")
            return model

        except Exception as e:
            print(f"[!] Training failed: {e}")
            return None

    def _find_last_checkpoint(self):
        runs_dir = Path("runs/detect")
        if not runs_dir.exists():
            return None
        subdirs = sorted(
            (d for d in runs_dir.iterdir() if d.is_dir()),
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        for d in subdirs:
            last_pt = d / "weights" / "last.pt"
            if last_pt.exists():
                return str(last_pt)
        return None

    def export_model(self, model=None, export_format="onnx"):
        if model is None:
            model = self._trained_model
        if model is None:
            best_pt = self._find_best_weights()
            if best_pt:
                print(f"[+] Loading best weights: {best_pt}")
                model = YOLO(best_pt)
            else:
                print("[!] No trained model found. Run --train first.")
                return False

        print(f"[+] Exporting to {export_format.upper()}...")
        try:
            exported_path = model.export(
                format=export_format,
                imgsz=self.config["imgsz"],
                opset=12,
                simplify=True,
            )
            exported = str(exported_path)
            if os.path.exists(exported):
                try:
                    shutil.copy(exported, "cs2head.onnx")
                except OSError as e:
                    print(f"[!] Failed to copy model to cs2head.onnx: {e}")
                    return False
                size_mb = os.path.getsize("cs2head.onnx") / (1024 * 1024)
                print(f"[+] Exported to cs2head.onnx ({size_mb:.1f} MB)")
                return True
            print("[!] Export produced no file")
            return False

        except Exception as e:
            print(f"[!] Export failed: {e}")
            return False

    def _find_best_weights(self):
        runs_dir = Path("runs/detect")
        if not runs_dir.exists():
            return None
        subdirs = sorted(
            (d for d in runs_dir.iterdir() if d.is_dir()),
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        for d in subdirs:
            best_pt = d / "weights" / "best.pt"
            if best_pt.exists():
                return str(best_pt)
        return None

    def validate_model(self, model_path="cs2head.onnx"):
        print(f"[+] Validating {model_path}...")
        if not os.path.exists(model_path):
            print(f"[!] Model not found: {model_path}")
            return False

        try:
            model = YOLO(model_path)

            test_images = find_images(f"{self.dataset_path}/images/val")[:5]
            if test_images:
                print("[+] Running inference on validation samples...")
                results = model(test_images)
                total_dets = 0
                for i, result in enumerate(results):
                    n = len(result.boxes) if result.boxes is not None else 0
                    total_dets += n
                    print(f"  Image {i+1}: {n} detections")
                print(f"[+] Total: {total_dets} detections across {len(test_images)} images")
            else:
                print("[+] No val images for testing, but model loads OK")
            return True

        except Exception as e:
            print(f"[!] Validation failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="CS2 Head Detection Model Training")
    parser.add_argument("--setup", action="store_true", help="Set up dataset structure")
    parser.add_argument("--split", action="store_true", help="Auto-split train into train+val")
    parser.add_argument("--check", action="store_true", help="Check dataset integrity")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--export", action="store_true", help="Export trained model to ONNX")
    parser.add_argument("--validate", action="store_true", help="Validate exported model")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--model", choices=["n", "s", "m", "l", "x"], default="n",
                        help="YOLO model size (default: n)")
    parser.add_argument("--device", default="0", help="GPU device ID or 'cpu' (default: 0)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Val split ratio for --split (default: 0.15)")

    args = parser.parse_args()

    trainer = CS2ModelTrainer()
    trainer.config["epochs"] = args.epochs
    trainer.config["batch"] = args.batch
    trainer.config["lr0"] = args.lr
    trainer.config["device"] = args.device
    trainer.config["resume"] = args.resume
    trainer.model_size = args.model

    run_any = args.setup or args.split or args.check or args.train or args.export or args.validate or args.all

    if not run_any:
        parser.print_help()
        return

    if args.setup or args.all:
        trainer.create_dataset_structure()

    if args.split or args.all:
        trainer.auto_split(val_ratio=args.val_ratio)

    if args.check or args.all:
        if not trainer.check_dataset():
            print("[!] Dataset check failed.")
            return

    if args.train or args.all:
        model = trainer.train_model()
        if not model:
            print("[!] Training failed!")
            return

    if args.export or args.all:
        if not trainer.export_model():
            print("[!] Export failed!")
            return

    if args.validate or args.all:
        trainer.validate_model()


if __name__ == "__main__":
    main()
