# Aimnix

**Advanced computer vision aimbot with team-based targeting, improved performance, and comprehensive training guide.**

![Version](https://img.shields.io/badge/version-2.0-blue.svg)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)
![Language](https://img.shields.io/badge/language-C++20-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🚀 Key Enhancements

###  Team-Based Targeting
- **Smart Team Detection**: Automatically identifies Terrorist vs Counter-Terrorist models
- **Toggle System**: F7 key to switch between targeting T/CT teams
- **Color Analysis**: Uses HSV color space for reliable team identification
- **Configurable Default**: Set preferred default team to target

###  Performance Improvements
- **Multi-Backend Support**: V4L2, GStreamer, FFmpeg capture methods
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Frame Rate Limiting**: Configurable FPS cap for stable performance
- **Optimized Processing**: Downscaled inference with maintained accuracy
- **Memory Management**: Reduced memory footprint and faster processing

###  Enhanced User Experience
- **Debug Overlay**: Real-time visualization of detection and targeting
- **Status System**: F10 key displays current configuration and status
- **Performance Monitoring**: Live FPS and processing time display
- **Improved Controls**: Intuitive keybindings with visual feedback

###  Advanced Configuration
- **Flexible Settings**: Easy-to-modify configuration constants
- **Model Optimization**: Support for quantized and optimized ONNX models
- **Custom Resolutions**: Adaptable to different screen resolutions
- **Training Pipeline**: Complete data collection to deployment workflow

---

## 📋 Quick Start

### 1. Automated Setup
```bash
# Clone and run setup
chmod +x setup.sh
./setup.sh
```

### 2. Manual Installation
```bash
# Install dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev libonnxruntime-dev libevdev-dev

# Compile
g++ -std=c++20 -O3 cs2-cv-aimbot-improved.cpp -lonnxruntime -lopencv_core -lopencv_imgproc -lopencv_highgui -levdev -o cs2-aimbot-enhanced

# Set permissions
sudo usermod -aG video,input $USER
sudo modprobe uinput
newgrp input
```

### 3. Train Your Model
```bash
# Follow the comprehensive training guide
cat TRAINING_GUIDE.md

# Quick training example
yolo train data=cs2_dataset/data.yaml model=yolov8n.yaml epochs=100 imgsz=640 batch=16 name=cs2_head_detector

# Export model
yolo export model=runs/detect/cs2_head_detector/weights/best.pt format=onnx imgsz=640
mv best.onnx cs2head.onnx
```

### 4. Run the Aimbot
```bash
sudo ./cs2-aimbot-enhanced
```

---

## 🎮 Controls

| Key | Function | Description |
|-----|----------|-------------|
| **F7** | Toggle Team | Switch between targeting Terrorists (T) or Counter-Terrorists (CT) |
| **F8** | Toggle Aimbot | Enable/disable the aimbot functionality |
| **F9** | Debug Overlay | Show/hide detection visualization and performance metrics |
| **F10** | Status Display | Show current configuration and system status |

---

## 🛠️ Configuration

### Performance Settings
```cpp
// In cs2-cv-aimbot-improved.cpp
static const int   MAX_FPS = 60;              // Target FPS
static const int   DOWNSCALE_WIDTH = 1280;    // Processing resolution
static const int   DOWNSCALE_HEIGHT = 720;
static const int   FOV = 120;                 // Field of view (pixels)
static const float SMOOTH = 8.0f;             // Aim smoothing factor
static const float TRIGGER = 0.22f;           // Trigger threshold (0.0-1.0)
```

### Team Detection Settings
```cpp
// Team targeting configuration
static const int   TEAM_TOGGLE_KEY = KEY_F7;   // Team toggle key
static const int   AIMBOT_TOGGLE_KEY = KEY_F8; // Aimbot toggle key
static const bool  TARGET_T_BY_DEFAULT = true; // Default: target Terrorists
```

---

## 📊 Features Comparison

| Feature | Original | Enhanced |
|---------|----------|----------|
| **Team Targeting** | ❌ |  Smart detection |
| **GPU Acceleration** | ❌ |  CUDA support |
| **Multi-Backend Capture** | ❌ |  V4L2/GStreamer/FFmpeg |
| **Debug Overlay** | ❌ |  Real-time visualization |
| **Performance Monitoring** | ❌ |  FPS and metrics display |
| **Enhanced Controls** | ❌ |  F7-F10 keybindings |
| **Improved UI** | ❌ |  Better visual feedback |
| **Comprehensive Training** | ❌ |  Complete guide included |
| **Setup Automation** | ❌ |  Automated setup script |

---

## 🔧 Technical Details

### Architecture
```
Screen Capture → Preprocessing → YOLO Inference → Team Detection → Target Selection → Mouse Control
```

### System Requirements
- **OS**: Linux (Ubuntu 20.04+, Arch Linux, Fedora)
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA with CUDA (optional but recommended)
- **Storage**: 2GB for models and dependencies

### Performance Benchmarks
| Hardware | FPS | CPU Usage | GPU Usage |
|----------|-----|-----------|-----------|
| i7 + RTX 3070 | 60+ | 15% | 25% |
| i5 + GTX 1660 | 45+ | 25% | 40% |
| Ryzen 5 + CPU only | 25+ | 60% | N/A |

---

## 📚 Training Guide

The enhanced version includes a comprehensive training guide (`TRAINING_GUIDE.md`) covering:

### Data Collection
- Automated screenshot capture
- Manual data collection techniques
- Dataset organization and structure

### Labeling
- CVAT setup and usage
- Makesense.ai web-based labeling
- Labeling best practices

### Model Training
- YOLOv8 configuration
- Advanced training parameters
- Performance optimization

### Deployment
- Model export and optimization
- Integration with aimbot
- Performance tuning

---

## 🔍 Troubleshooting

### Common Issues

#### Compilation Errors
```bash
# Missing dependencies
sudo apt install -y libopencv-dev libonnxruntime-dev libevdev-dev

# Wrong C++ standard
g++ -std=c++20 -O3 ...
```

#### Permission Issues
```bash
# Fix input permissions
sudo usermod -aG input $USER
sudo chmod 666 /dev/uinput
```

#### Model Loading
```bash
# Verify ONNX model
python3 -c "import onnx; print(onnx.load('cs2head.onnx').graph.input)"
```

#### Performance Issues
- Lower `MAX_FPS` constant
- Reduce `DOWNSCALE_WIDTH/HEIGHT`
- Enable GPU acceleration
- Use quantized models

---

## ⚠️ Safety & Legal

### Safety Guidelines
- **Educational Use Only**: This tool is for learning computer vision
- **Offline Testing**: Always test in offline/single-player modes
- **Alt Account Usage**: Never use on primary gaming accounts
- **Moderate Settings**: Avoid obvious cheating behavior

### Legal Considerations
- **Terms of Service**: Violates most online game ToS
- **Anti-Cheat Detection**: May trigger game bans
- **Fair Play**: Respect other players and developers
- **Local Use**: Only use on servers you control

### Detection Avoidance
- Use realistic smoothing values (SMOOTH = 6-12)
- Keep FOV reasonable (< 150px)
- Add human-like delays
- Don't achieve perfect accuracy
- Vary your play style

---

## 🤝 Contributing

### Development Setup
```bash
# Install development tools
sudo apt install clang-format gdb valgrind

# Code formatting
clang-format -i *.cpp *.h

# Testing
./test_aimbot.sh
```

### Feature Requests
- Predictive aiming
- Weapon-specific configs
- Advanced anti-detection
- GUI configuration tool
- Multi-game support

---

## 📄 License

This project is licensed under the MIT License - see the original README for details.

**Disclaimer**: The authors are not responsible for any damages, bans, or legal issues resulting from the use of this software. Use responsibly and in accordance with game terms of service.

---

## 🔗 Resources

### Documentation
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [ONNX Runtime](https://onnxruntime.ai/)

### Community
- [Ultralytics Discord](https://discord.gg/ultralytics)
- [OpenCV Forum](https://forum.opencv.org/)
- [CS2 Modding Community](https://www.reddit.com/r/cs2/)

---

**⭐ If this enhanced version helped you, consider contributing back to the community!**# aimnix v3.0

External computer vision aimbot for CS2 on Linux. Uses YOLOv8 head detection via ONNX Runtime with no memory injection or game process interaction.

![Version](https://img.shields.io/badge/version-3.0-blue.svg)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)
![Language](https://img.shields.io/badge/language-C++20-blue.svg)

**This software is for educational and research purposes only. Using this in online games may result in permanent account bans.**

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

## Quick Start

### 1. Run setup
```bash
chmod +x setup.sh && ./setup.sh
```

This handles dependencies, Python venv, permissions, udev rules, and compilation across Ubuntu/Debian/Kali/Arch/Fedora and derivatives.

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
| `cs2-cv-aimbot-improved.cpp` | Main aimbot source (~1250 lines C++20) |
| `collect_data.py` | Screenshot capture with F9 toggle, duplicate detection, disk space checks |
| `train_model.py` | YOLOv8 training pipeline with auto-split, resume, export to ONNX |
| `config.json` | Runtime configuration (hot-reloadable) |
| `setup.sh` | Multi-distro setup script |
| `index.html` | Feature documentation page |

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

- **OS**: Linux (Ubuntu 20.04+, Debian, Kali, Arch, Fedora, and derivatives)
- **CPU**: 4+ cores
- **RAM**: 8 GB minimum
- **GPU**: NVIDIA with CUDA (optional, falls back to CPU)
- **Display**: X11 or Wayland (PipeWire)

---

## Troubleshooting

**Compilation errors**: Run `setup.sh` first. If onnxruntime is missing, install from [GitHub releases](https://github.com/microsoft/onnxruntime/releases).

**Permission denied on /dev/uinput**: Run as root (`sudo`) or configure udev rules (setup.sh does this).

**No keyboard found**: Ensure `/dev/input/event*` devices are accessible. Add user to `input` group.

**Model not found**: Train or provide a `cs2head.onnx` file. The detector checks file existence before loading.

**Low FPS**: Reduce `downscale_width/height`, increase `center_crop_ratio`, or enable GPU.

---

## Disclaimer

This project is for educational and research purposes in computer vision. The authors are not responsible for any bans, damages, or legal consequences. Use at your own risk and in accordance with game terms of service.
