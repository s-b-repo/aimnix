# CS2 Head Detection Model Training Guide

This guide covers the complete training pipeline for the aimnix CS2 CV aimbot, from data collection to model export.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Data Collection](#step-1-data-collection)
3. [Step 2: Data Labeling](#step-2-data-labeling)
4. [Step 3: Dataset Preparation](#step-3-dataset-preparation)
5. [Step 4: Model Training](#step-4-model-training)
6. [Step 5: Model Export](#step-5-model-export)
7. [Step 6: Model Validation](#step-6-model-validation)
8. [Complete Pipeline](#complete-pipeline)
9. [Performance Optimization](#performance-optimization)

---

## Prerequisites

### System Requirements
- Linux (Ubuntu 20.04+, Arch Linux, EndeavourOS, Fedora, or derivatives)
- 4+ CPU cores, 8GB+ RAM
- NVIDIA GPU with CUDA (recommended for training speed)
- 2GB+ free disk space for models and dependencies

### Installed Dependencies
The `setup.sh` script handles system-level dependencies (OpenCV, libevdev, ONNX Runtime, GStreamer, CUDA). It also creates a Python virtual environment at `cs2_training_env/` with all training dependencies.

If you need to set up the Python environment manually:
```bash
cd aimnix
python3 -m venv cs2_training_env
source cs2_training_env/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python numpy pillow mss keyboard
```

---

## Step 1: Data Collection

### Method 1: Automated Screenshot Capture (Recommended)

Use the built-in `collect_data.py` script:
```bash
source cs2_training_env/bin/activate
python3 collect_data.py --rate 5 --duration 10
```

Options:
- `--rate` / `-r`: Capture rate in FPS (default: 5)
- `--duration` / `-d`: Max duration in minutes, 0 = unlimited (default: 0)
- `--quality` / `-q`: JPEG quality 1-100 (default: 85)
- `--output` / `-o`: Output directory (default: `cs2_dataset/images/train`)
- `--region`: Capture region as x,y,w,h (default: full screen)
- `--val-split`: Fraction of frames to val/ (default: 0.15)

Controls:
- **F9**: Start/stop capture
- **ESC**: Exit

The script automatically:
- Detects duplicate frames (via hash) and skips them
- Checks available disk space
- Splits data into train/val directories
- Names files with timestamps for uniqueness

### Method 2: FFmpeg Screen Recording

```bash
# Record at 30 FPS, then extract frames
ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :0 \
    -vf fps=5 cs2_dataset/raw/frame_%04d.jpg
```

### Labeling Guidelines
- Capture diverse scenarios: different maps, angles, distances, lighting
- Aim for 500-1000+ images of enemy head visibility
- Include some non-target frames (no enemies visible) for negative samples

---

## Step 2: Data Labeling

### Option A: MakeSense.ai (Web-based, Easiest)

1. Visit [makesense.ai](https://www.makesense.ai/)
2. Upload your collected images from `cs2_dataset/images/train/`
3. Create a "head" label
4. Draw bounding boxes around every visible enemy head
5. Export as **YOLO format**
6. Download and extract into `cs2_dataset/labels/train/`

### Option B: CVAT (Local Installation)

```bash
docker pull cvat/server
docker run -d -p 8080:8080 cvat/server
# Access at http://localhost:8080
```

1. Create a new project, upload images
2. Create "head" label
3. Annotate all frames
4. Export as YOLO format

### Labeling Best Practices
- Include the entire head with a small margin
- Label partially visible heads (they train the model on occlusion)
- Be consistent — similar head positions = similar box sizes
- Don't label friendly players (only enemies)

---

## Step 3: Dataset Preparation

The training script auto-generates the YAML config and dataset structure. You can also do it manually:

```yaml
# cs2_dataset/data.yaml
path: /home/cortix/Templates/aimnix/cs2_dataset
train: images/train
val: images/val
test: images/test

nc: 1  # number of classes
names: ['head']
```

The script creates this automatically. Directory structure:
```
cs2_dataset/
  data.yaml
  images/
    train/   <- annotated training images
    val/     <- validation images (auto-split if empty)
    test/    <- test images
  labels/
    train/   <- YOLO .txt annotations
    val/     <- validation annotations
    test/    <- test annotations
```

### Manual Split
```python
# Auto-split is built into train_model.py --split
# Manual alternative:
from train_model import CS2ModelTrainer
trainer = CS2ModelTrainer()
trainer.auto_split(val_ratio=0.15, seed=42)
```

---

## Step 4: Model Training

### Quick Training
```bash
source cs2_training_env/bin/activate
python3 train_model.py --train --epochs 100 --model n
```

### Full Pipeline (Setup + Split + Check + Train + Export + Validate)
```bash
python3 train_model.py --all --epochs 100
```

### Command Options
| Option | Description | Default |
|--------|-------------|---------|
| `--epochs` | Number of training epochs | 100 |
| `--model` | YOLO model size: n/s/m/l/x | n |
| `--device` | GPU device ID or "cpu" | 0 |
| `--batch` | Batch size | 16 |
| `--lr` | Initial learning rate | 0.01 |
| `--resume` | Resume from last checkpoint | off |
| `--val-ratio` | Val split ratio for `--split` | 0.15 |
| `--setup` | Create dataset structure | off |
| `--split` | Auto-split train into train+val | off |
| `--check` | Check dataset integrity | off |
| `--train` | Train the model | off |
| `--export` | Export to ONNX | off |
| `--validate` | Validate exported model | off |
| `--all` | Run full pipeline | off |

### Model Size Recommendations
- **n (nano)**: Fastest, smallest, good for CPU. Use if GPU not available.
- **s (small)**: Good balance of speed and accuracy.
- **m (medium)**: Better accuracy, more compute.
- **l/x (large/xlarge)**: Best accuracy, very slow training.

### Monitoring Training
```bash
tensorboard --logdir runs/detect
# Open browser to http://localhost:6006
```

---

## Step 5: Model Export

The `--export` flag (or `--all`) automatically exports the best trained weights to ONNX:

```bash
python3 train_model.py --export
# Output: cs2head.onnx (YOLO format, opset 12, simplified)
```

This will:
1. Find the best weights from the latest training run
2. Export to ONNX format with imgsz=640
3. Copy the result to `cs2head.onnx` in the project root

The aimbot binary automatically loads `cs2head.onnx` at startup.

### Manual Export
```python
from ultralytics import YOLO
model = YOLO("runs/detect/cs2_head_n/weights/best.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
# Output: best.onnx
import shutil; shutil.copy("best.onnx", "cs2head.onnx")
```

---

## Step 6: Model Validation

```bash
python3 train_model.py --validate
```

This checks:
- ONNX model loads correctly
- Runs inference on validation samples
- Reports detection counts for sanity checking

---

## Complete Pipeline

The simplest way to train from scratch:

```bash
# 1. Collect data (in CS2 game)
source cs2_training_env/bin/activate
python3 collect_data.py --rate 5 --duration 15

# 2. Label data externally (makesense.ai), save labels to cs2_dataset/labels/train/

# 3. Train and export (one command)
python3 train_model.py --all --epochs 100 --model n

# 4. Run the aimbot
sudo ./cs2-aimbot-enhanced
```

---

## Performance Optimization

### ONNX Runtime Inference
- The aimbot uses ONNX Runtime for inference
- CUDA execution provider is used automatically if a GPU is available
- Falls back to CPU if CUDA is not available
- Center-screen cropping (`center_crop_ratio` in config.json) reduces input size for faster inference

### Model Quantization (Advanced)
```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize for smaller size and faster inference
quantize_dynamic(
    "cs2head.onnx",
    "cs2head_quantized.onnx",
    weight_type=QuantType.QUInt8
)
# Use cs2head_quantized.onnx as model_path in config.json
```

### Training Tips
- Start with `--model n` for quick iteration
- 100 epochs is a good default; increase if accuracy is insufficient
- Use data augmentation (enabled by default in the script)
- Ensure diverse training data (different maps, lighting, angles)
- Check val/ detection results with `--validate`

---

## Troubleshooting

### Training Fails: "CUDA out of memory"
- Reduce `--batch` size (try 8 or 4)
- Use `--device cpu` for CPU-only training
- Use a smaller model (`--model n`)

### Training Fails: "No training images found"
- Ensure images are in `cs2_dataset/images/train/`
- Run `python3 train_model.py --setup` to create the structure
- Run `python3 train_model.py --split` to auto-split from train to val

### Exporting Fails
- Ensure training completed and `runs/detect/*/weights/best.pt` exists
- Check disk space
- Try `--model n` for smaller export

### Model Not Detecting
- Check that bounding boxes in labels are correct
- Lower `confidence_threshold` in config.json
- Retrain with more diverse data
- Verify the model loads: `python3 -c "from ultralytics import YOLO; YOLO('cs2head.onnx')"`