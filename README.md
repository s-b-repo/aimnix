# Aimnix — CS2 Computer Vision Aimbot v3.0

**External computer vision aimbot for CS2 on Linux. Uses YOLOv8 head detection via ONNX Runtime with no memory injection or game process interaction.**

![Version](https://img.shields.io/badge/version-3.0-blue.svg)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)
![Language](https://img.shields.io/badge/language-C++20-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Controls](#controls)
- [Configuration](#configuration)
- [Training](#training)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Run setup
```bash
chmod +x setup.sh
./setup.sh
```
This handles dependencies, Python venv, permissions, udev rules, and compilation across Ubuntu/Debian/Kali/Arch/Fedora and derivatives.

> **Note:** `setup.sh` requires `sudo` for installing system packages and configuring udev rules. If sudo is not available or passwordless, you can install dependencies and compile manually (see [Manual Setup](#manual-setup) below).

### 2. Collect training data
```bash
source cs2_training_env/bin/activate
python3 collect_data.py --rate 5 --duration 10
```
Press F9 to start/stop capture. ESC to exit.

### 3. Label data
Use [makesense.ai](https://www.makesense.ai/) or CVAT. Export as YOLO format into `cs2_dataset/labels/train/`.

### 4. Train model
```bash
python3 train_model.py --all --epochs 100
```
Options: `--model n/s/m/l/x`, `--device cpu`, `--batch 16`, `--resume`, `--split`.

### 5. Run
```bash
sudo ./cs2-aimbot-enhanced
```
Optional: `sudo ./cs2-aimbot-enhanced /path/to/config.json`

---

## Features

### Aim Engine
- YOLOv8 head detection via ONNX Runtime (GPU or CPU)
- Assist mode (smooth tracking) and Lock mode (snap while key held)
- Cubic ease-out smoothing curve with dynamic factor
- Dead zone to suppress micro-adjustments near crosshair
- Weighted target scoring combining distance and confidence
- Configurable FOV radius

### Stealth System
- Gaussian micro-jitter on aim path
- Random overshoot with sinusoidal correction
- Per-frame aim speed variance
- Per-target aim offset randomization within bounding box
- Target switch penalty (stickiness to current target)
- Randomized reaction delay on new target acquisition
- Configurable miss probability
- Fire rate limiter (minimum interval between shots)
- Random process name masking from a pool of 27 common Linux process names (htop, konsole, plasmashell, etc.)

### Anti-Recoil
- Automatic downward compensation during sustained fire
- Ramps up strength over burst duration
- Randomized horizontal drift per spray
- Gaussian noise on both axes
- Periodic micro-pauses in compensation for human-like variance

### Team Detection
- HSV color analysis on body region below the head (not head itself)
- Pixel-level red/blue classification with saturation and value thresholds
- T-side: hue 0-25 and 155-180 (reds, oranges)
- CT-side: hue 95-135 (blues)
- Targets unknown-team players by default

### Performance
- Center-screen cropping for faster inference (configurable ratio)
- CUDA GPU acceleration with automatic fallback to CPU
- Deadline-based frame timing (subtracts processing time from sleep)
- GStreamer capture: ximagesrc (X11) or pipewiresrc (Wayland)
- Configurable downscale resolution

### Extra
- Bunny hop auto-jump via virtual keyboard (uinput)
- Status overlay on secondary monitor (OpenCV window)
- Config hot-reload at runtime (F11)
- Panic kill-switch (F12) disables everything instantly
- Auto train/val dataset splitting

---

## Controls

| Key | Function | Description |
|-----|----------|-------------|
| F5 | Aim Mode | Toggle between Assist (smooth) and Lock (snap) |
| F6 | Bunny Hop | Toggle auto-jump |
| F7 | Team | Switch target team (T / CT) |
| F8 | Aimbot | Enable / disable aimbot |
| F9 | Debug | Toggle detection overlay |
| F10 | Status | Show all current states (5s) |
| F11 | Reload | Hot-reload config.json |
| F12 | Panic | Kill switch - disable everything |
| Left Alt | Aim Lock | Hold to snap in Lock mode (aim_mode=1) |

---

## Configuration

All settings are in `config.json` and loaded at startup. Press F11 to hot-reload without restarting.

### Sections

| Section | Key Settings |
|---------|-------------|
| `aimbot` | fov, smooth, trigger_threshold, max_fps, aim_mode, dead_zone, bhop_enabled |
| `controls` | Key bindings for all toggles (F5-F12) |
| `display` | debug_overlay, show_fps, overlay_monitor |
| `performance` | use_gpu, gpu_device, cpu_threads, center_crop_ratio |
| `detection` | model_path, input_size, nms_threshold, color_threshold |
| `safety` | max_aim_speed, reaction_delay_min/max, miss_probability |
| `recoil` | enabled, strength, delay_ms |
| `stealth` | jitter_intensity, overshoot_chance/amount, speed_variance, min_fire_interval_ms, aim_offset_range, target_switch_penalty |

See `config.json` for all defaults and valid ranges.

---

## Files

| File | Description |
|------|-------------|
| `cs2-cv-aimbot-improved.cpp` | Main aimbot source (~1373 lines C++20) |
| `collect_data.py` | Screenshot capture with F9 toggle, duplicate detection, disk space checks |
| `train_model.py` | YOLOv8 training pipeline with auto-split, resume, export to ONNX |
| `config.json` | Runtime configuration (hot-reloadable) |
| `setup.sh` | Multi-distro setup script |
| `index.html` | Feature documentation page |
| `cs2head.onnx` | Pre-trained YOLOv8 head detection model |
| `third_party/onnxruntime/` | Bundled ONNX Runtime headers and library (v1.28.0) |

---

## Architecture

```
Screen Capture (GStreamer)
    -> Center Crop
    -> Downscale
    -> YOLOv8 ONNX Inference
    -> NMS + Confidence Filter
    -> Team Color Detection (HSV body sampling)
    -> Target Scoring (distance * confidence * stickiness)
    -> Aim Smoothing (cubic ease-out + stealth jitter/overshoot)
    -> Anti-Recoil Compensation
    -> uinput Virtual Mouse Output
```

### Key Classes
- `Config` - JSON parser, validation, clamping, process name pool
- `UMouse` - uinput virtual mouse (relative moves + clicks)
- `BhopEngine` - uinput virtual keyboard for auto-jump
- `AntiRecoil` - Recoil compensation with ramp, drift, and noise
- `AdvancedToggle` - libevdev keyboard input, F-key handling
- `StatusOverlay` - Secondary monitor overlay (thread-safe)
- `ScreenCapture` - GStreamer X11/Wayland capture
- `HeadDetector` - ONNX Runtime YOLOv8 inference
- `TeamDetector` - HSV body-region color classification
- `StealthEngine` - Jitter, overshoot, offsets, fire rate
- `AimSmoother` - Cubic ease-out with dynamic factor
- `CS2Aimbot` - Main loop tying everything together

---

## System Requirements

- **OS**: Linux (Ubuntu 20.04+, Debian, Kali, Arch, Fedora, EndeavourOS, and derivatives)
- **CPU**: 4+ cores
- **RAM**: 8 GB minimum
- **GPU**: NVIDIA with CUDA (optional, falls back to CPU)
- **Display**: X11 or Wayland (PipeWire)

---

## Manual Setup

### Arch / EndeavourOS / Manjaro
```bash
# Install system dependencies
sudo pacman -S --noconfirm base-devel cmake pkgconf opencv libevdev gst-plugins-good onnxruntime-cuda cuda

# Create Python venv
python3 -m venv cs2_training_env
source cs2_training_env/bin/activate
pip install ultralytics opencv-python numpy pillow mss keyboard

# Create dataset directories
mkdir -p cs2_dataset/{images/{train,val,test},labels/{train,val,test}}

# Compile (link specific OpenCV modules to avoid VTK/HDF5 deps)
g++ -std=c++20 -O3 cs2-cv-aimbot-improved.cpp \
    $(pkg-config --cflags opencv5) $(pkg-config --cflags libevdev) \
    -I third_party/onnxruntime/include -L third_party/onnxruntime/lib \
    -lonnxruntime -lopencv_core -lopencv_imgproc -lopencv_highgui \
    -lopencv_dnn -lopencv_videoio -levdev -lpthread \
    -Wl,-rpath,third_party/onnxruntime/lib \
    -o cs2-aimbot-enhanced

# Set up permissions
sudo usermod -aG video,input $USER
sudo modprobe uinput
sudo udevadm control --reload-rules
newgrp input

# Run
sudo ./cs2-aimbot-enhanced
```

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev libonnxruntime-dev libevdev-dev libgstreamer1.0-dev gstreamer1.0-plugins-good nvidia-cuda-toolkit python3 python3-pip python3-venv
```

### Fedora
```bash
sudo dnf install -y gcc-c++ cmake pkgconf-pkg-config opencv-devel onnxruntime-devel libevdev-devel gstreamer1-plugins-good cuda-toolkit python3 python3-pip
```

---

## Troubleshooting

**Compilation errors**: Run `setup.sh` first. If onnxruntime is missing, install from [GitHub releases](https://github.com/microsoft/onnxruntime/releases) and place headers/libs in `third_party/onnxruntime/`.

**Permission denied on /dev/uinput**: Run as root (`sudo`) or configure udev rules (setup.sh does this).

**No keyboard found**: Ensure `/dev/input/event*` devices are accessible. Add user to `input` group.

**Model not found**: Train or provide a `cs2head.onnx` file. The detector checks file existence before loading.

**Low FPS**: Reduce `downscale_width/height`, increase `center_crop_ratio`, or enable GPU.

**OpenCV link errors**: The system package may include VTK/HDF5 modules that need extra deps. Link only the specific modules needed: `-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_dnn -lopencv_videoio`.

**GPU not detected**: Ensure NVIDIA drivers and CUDA toolkit are installed. The binary falls back to CPU inference automatically.

---

## Disclaimer

This project is for educational and research purposes in computer vision. The authors are not responsible for any bans, damages, or legal consequences. Use at your own risk and in accordance with game terms of service.
