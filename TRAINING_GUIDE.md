# CS2 CV Aimbot - Complete Training Guide

## Overview
This enhanced CS2 Computer Vision aimbot includes team-based targeting, improved performance, and comprehensive debug features. The system uses machine learning to detect enemy heads and can distinguish between Terrorist and Counter-Terrorist teams.

## 🔧 Enhanced Features

### Team Toggle System
- **F7 Key**: Toggle between targeting Terrorists (T) or Counter-Terrorists (CT)
- **Smart Targeting**: Uses color analysis to identify team affiliation
- **Default Behavior**: Configurable default team to target

### Performance Improvements
- **Optimized Screen Capture**: Multiple backend support (V4L2, GStreamer, FFmpeg)
- **GPU Acceleration**: Automatic CUDA detection and usage
- **Frame Rate Limiting**: Configurable FPS cap (default 60 FPS)
- **Downscaled Processing**: Faster inference with maintained accuracy

### Debug & Monitoring
- **Real-time FPS Display**: Performance monitoring
- **Team Detection Overlay**: Visual feedback for team identification
- **Status Indicators**: Current targeting mode and system state
- **Enhanced Visual Feedback**: Different colors for different teams

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+, Arch Linux, or similar)
- **CPU**: Modern multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 2GB free space for models and dependencies

### Dependencies Installation

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libonnxruntime-dev \
    libevdev-dev \
    libcuda-dev \
    nvidia-cuda-toolkit
```

#### Arch Linux
```bash
sudo pacman -S \
    base-devel \
    cmake \
    opencv \
    onnxruntime \
    libevdev \
    cuda
```

### Python Environment Setup
```bash
# Create virtual environment
python3 -m venv cs2_training_env
source cs2_training_env/bin/activate

# Install training dependencies
pip install --upgrade pip
pip install \
    ultralytics \
    opencv-python \
    numpy \
    pillow \
    matplotlib \
    tensorboard \
    roboflow
```

## 🎯 Training Your Custom Model

### Step 1: Data Collection

#### Method 1: Automated Screenshot Capture
```bash
# Create dataset structure
mkdir -p cs2_dataset/{images,labels}

# Start automated capture (run while playing CS2)
ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :0 \
    -vf fps=30 -q:v 2 cs2_dataset/images/frame_%04d.jpg
```

#### Method 2: Manual Screenshot Collection
1. Play CS2 in windowed/borderless mode
2. Take screenshots when enemies are visible
3. Save to `cs2_dataset/images/` folder
4. Aim for 500-1000 diverse images

### Step 2: Data Labeling

#### Using CVAT (Recommended)
```bash
# Install CVAT
docker pull cvat/server
docker run -d -p 8080:8080 cvat/server

# Access at http://localhost:8080
# Upload images and label heads with bounding boxes
# Export as YOLO format
```

#### Using Makesense.ai (Web-based)
1. Visit https://www.makesense.ai/
2. Upload your images
3. Label heads with bounding boxes
4. Export in YOLO format
5. Save labels to `cs2_dataset/labels/`

#### Labeling Guidelines
- **Label Only Heads**: Focus on the head region
- **Consistent Bounding**: Include entire head with some margin
- **Multiple Angles**: Include different viewing angles
- **Various Distances**: Near, medium, and far targets
- **Different Maps**: Train on multiple CS2 maps

### Step 3: Dataset Preparation

#### Create Data Configuration
```yaml
# cs2_dataset/data.yaml
path: ./cs2_dataset
train: images/train
val: images/val
test: images/test

nc: 1  # number of classes
names: ['head']  # class names
```

#### Split Dataset
```python
import os
import shutil
import random

def split_dataset(source_dir, train_ratio=0.8, val_ratio=0.1):
    images_dir = os.path.join(source_dir, 'images')
    labels_dir = os.path.join(source_dir, 'labels')
    
    # Get all image files
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(images)
    
    # Calculate split indices
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(source_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(source_dir, 'labels', split), exist_ok=True)
    
    # Split files
    for i, image in enumerate(images):
        label = image.replace('.jpg', '.txt').replace('.png', '.txt')
        
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'
        
        # Move image and label
        shutil.move(os.path.join(images_dir, image),
                   os.path.join(source_dir, 'images', split, image))
        if os.path.exists(os.path.join(labels_dir, label)):
            shutil.move(os.path.join(labels_dir, label),
                       os.path.join(source_dir, 'labels', split, label))

split_dataset('cs2_dataset')
```

### Step 4: Model Training

#### Basic Training
```bash
# Train YOLOv8n model
yolo train \
    data=cs2_dataset/data.yaml \
    model=yolov8n.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    name=cs2_head_detector \
    device=0  # GPU device ID
```

#### Advanced Training Options
```bash
# With augmentation and optimization
yolo train \
    data=cs2_dataset/data.yaml \
    model=yolov8n.yaml \
    epochs=150 \
    imgsz=640 \
    batch=32 \
    name=cs2_head_detector_advanced \
    device=0 \
    augment=True \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    degrees=10 \
    translate=0.1 \
    scale=0.5 \
    shear=10 \
    perspective=0.0 \
    flipud=0.0 \
    fliplr=0.5 \
    mosaic=1.0 \
    mixup=0.0 \
    copy_paste=0.0
```

#### Monitor Training
```bash
# View training progress
tensorboard --logdir runs/detect
# Open browser to http://localhost:6006
```

### Step 5: Model Export

#### Export to ONNX
```bash
# Export best model
yolo export \
    model=runs/detect/cs2_head_detector/weights/best.pt \
    format=onnx \
    imgsz=640 \
    opset=12 \
    optimize=True

# Copy to aimbot directory
cp runs/detect/cs2_head_detector/weights/best.onnx cs2head.onnx
```

#### Optimize Model (Optional)
```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load and optimize model
model = onnx.load("cs2head.onnx")
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "cs2head_optimized.onnx")

# Quantize for better performance
quantize_dynamic(
    "cs2head_optimized.onnx",
    "cs2head_quantized.onnx",
    weight_type=QuantType.QUInt8
)
```

## 🚀 Running the Enhanced Aimbot

### Compilation
```bash
g++ -std=c++20 -O3 cs2-cv-aimbot-improved.cpp \
    -lonnxruntime \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_highgui \
    -levdev \
    -o cs2-aimbot-enhanced
```

### System Setup
```bash
# Set up permissions (one-time setup)
sudo usermod -aG video,input $USER
sudo modprobe uinput
sudo udevadm control --reload-rules

# Reload groups (or restart system)
newgrp input
```

### Launch Options
```bash
# Basic launch
sudo ./cs2-aimbot-enhanced

# With custom model
sudo ./cs2-aimbot-enhanced --model=custom_model.onnx

# With debug mode
sudo ./cs2-aimbot-enhanced --debug
```

## 🎮 In-Game Controls

| Key | Function |
|-----|----------|
| **F7** | Toggle target team (T/CT) |
| **F8** | Enable/disable aimbot |
| **F9** | Toggle debug overlay |
| **F10** | Show current status |

### Team Targeting Behavior
- **T Mode**: Prioritizes terrorist-colored models (red/orange)
- **CT Mode**: Prioritizes counter-terrorist-colored models (blue)
- **Unknown Teams**: Targets all detected heads when team detection is uncertain

## 📊 Performance Optimization

### System Tuning
```bash
# CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase GPU performance (NVIDIA)
nvidia-smi -pm 1
nvidia-smi -acp 0
nvidia-smi -i 0 -ac 5000,1700

# Optimize system for gaming
echo 1 | sudo tee /proc/sys/kernel/sched_rt_runtime_us
```

### Game Settings
- **Resolution**: Match capture resolution (1280x720 recommended)
- **Display Mode**: Borderless Window
- **V-Sync**: Disabled
- **Anti-Aliasing**: Low or Off
- **Shadows**: Low (improves head visibility)

## 🔍 Troubleshooting

### Common Issues

#### Permission Denied
```bash
# Fix uinput permissions
sudo chmod 666 /dev/uinput
sudo usermod -aG input $USER
```

#### No Screen Capture
```bash
# Check available devices
ls /dev/video*
v4l2-ctl --list-devices

# Test capture
cv2.VideoCapture(0).read()
```

#### Model Not Loading
```bash
# Verify ONNX model
python3 -c "
import onnx
model = onnx.load('cs2head.onnx')
print(model.graph.input)
print(model.graph.output)
"
```

#### Poor Detection Accuracy
1. **Retrain Model**: Collect more diverse data
2. **Adjust Confidence**: Lower detection threshold
3. **Check Lighting**: Ensure consistent lighting conditions
4. **Update Model**: Use newer YOLO architecture

### Debug Mode
Enable debug overlay (F9) to see:
- Real-time detection boxes
- Team classification results
- Performance metrics
- Target selection logic

## ⚠️ Legal and Safety Information

### Terms of Use
- **Educational Purpose**: This tool is for educational and research purposes only
- **Game Violations**: Using this in online games may result in permanent bans
- **Fair Play**: Respect other players and game developers
- **Local Use**: Only use on local servers or with permission

### Safety Guidelines
1. **Test Offline**: Always test in offline mode first
2. **Use Alt Account**: Never use on primary gaming account
3. **Moderate Settings**: Avoid obvious cheating behavior
4. **Stay Updated**: Keep up with game anti-cheat developments

### Detection Avoidance
- **Human-like Behavior**: Use realistic smoothing values
- **Limited FOV**: Keep FOV reasonable (under 150px)
- **Occasional Misses**: Don't achieve perfect accuracy
- **Random Delays**: Add human-like reaction delays

## 📈 Advanced Configuration

### Customizing Behavior
Edit the configuration section in `cs2-cv-aimbot-improved.cpp`:

```cpp
// Performance settings
static const int   MAX_FPS = 60;           // Target FPS
static const int   DOWNSCALE_WIDTH = 1280;  // Processing resolution
static const int   DOWNSCALE_HEIGHT = 720;

// Detection settings
static const float DETECTION_CONFIDENCE = 0.35f;
static const int   FOV = 120;              // Field of view in pixels

// Aim settings
static const float SMOOTH = 8.0f;          // Smoothing factor
static const float TRIGGER = 0.22f;        // Trigger threshold
```

### Creating Custom Models
```python
# Advanced training script
from ultralytics import YOLO
import yaml

# Load model
model = YOLO('yolov8n.yaml')

# Custom training with callbacks
results = model.train(
    data='cs2_dataset/data.yaml',
    epochs=200,
    imgsz=640,
    batch=32,
    name='cs2_advanced',
    callbacks={
        'on_train_epoch_end': lambda trainer: print(f"Epoch {trainer.epoch} completed"),
        'on_fit_epoch_end': lambda trainer: print(f"Metrics: {trainer.metrics}")
    }
)
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/cs2-cv-aimbot-enhanced
cd cs2-cv-aimbot-enhanced

# Install development dependencies
sudo apt install clang-format gdb valgrind

# Code style
clang-format -i *.cpp *.h
```

### Feature Requests
- Multi-target tracking
- Predictive aiming
- Weapon-specific configurations
- Advanced anti-detection methods
- GUI configuration tool

## 📚 Additional Resources

### YOLO Documentation
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [YOLO Training Tips](https://docs.ultralytics.com/guides/yolo-common-issues/)
- [Model Optimization](https://docs.ultralytics.com/modes/export/)

### OpenCV Resources
- [OpenCV Documentation](https://docs.opencv.org/)
- [Computer Vision Tutorials](https://opencv-tutorial.readthedocs.io/)

### CS2 Resources
- [CS2 Console Commands](https://developer.valvesoftware.com/wiki/Console_Command_List)
- [Game Optimization Guide](https://steamcommunity.com/sharedfiles/filedetails/?id=3002415546)

---

**Disclaimer**: This software is provided as-is for educational purposes. The authors are not responsible for any damages, bans, or legal issues resulting from the use of this software. Use responsibly and in accordance with game terms of service.