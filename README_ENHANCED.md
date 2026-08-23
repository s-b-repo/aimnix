# Aimnix README_ENHANCED

**Advanced computer vision aimbot with team-based targeting, improved performance, and comprehensive training guide.**

![Version](https://img.shields.io/badge/version-3.0-blue.svg)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)
![Language](https://img.shields.io/badge/language-C++20-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Key Enhancements

### Team-Based Targeting
- **Smart Team Detection**: Automatically identifies Terrorist vs Counter-Terrorist models
- **Toggle System**: F7 key to switch between targeting T/CT teams
- **Color Analysis**: Uses HSV color space for reliable team identification
- **Configurable Default**: Set preferred default team to target

### Performance Improvements
- **Multi-Backend Support**: V4L2, GStreamer, FFmpeg capture methods
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Frame Rate Limiting**: Configurable FPS cap for stable performance
- **Optimized Processing**: Downscaled inference with maintained accuracy
- **Memory Management**: Reduced memory footprint and faster processing

### Enhanced User Experience
- **Debug Overlay**: Real-time visualization of detection and targeting
- **Status System**: F10 key displays current configuration and status
- **Performance Monitoring**: Live FPS and processing time display
- **Improved Controls**: Intuitive keybindings with visual feedback

### Advanced Configuration
- **Flexible Settings**: Easy-to-modify configuration constants
- **Model Optimization**: Support for quantized and optimized ONNX models
- **Custom Resolutions**: Adaptable to different screen resolutions
- **Training Pipeline**: Complete data collection to deployment workflow

---

## Quick Start

```bash
# Clone and run setup
chmod +x setup.sh
./setup.sh

# Train a model (follow TRAINING_GUIDE.md)
source cs2_training_env/bin/activate
python3 train_model.py --all

# Run the enhanced aimbot
sudo ./cs2-aimbot-enhanced
```

### In-Game Controls

| Key | Function | Description |
|-----|----------|-------------|
| F7 | Toggle Team | Switch between targeting T/CT teams |
| F8 | Toggle Aimbot | Enable/disable aimbot |
| F9 | Debug Overlay | Show/hide detection visualization |
| F10 | Status Display | Show current configuration and system status |

---

## Features Comparison

| Feature | Original | Enhanced |
|---------|----------|----------|
| Team Targeting | No | Smart detection |
| GPU Acceleration | No | CUDA support |
| Multi-Backend Capture | No | V4L2/GStreamer/FFmpeg |
| Debug Overlay | No | Real-time visualization |
| Performance Monitoring | No | FPS and metrics display |
| Enhanced Controls | No | F7-F10 keybindings |
| Improved UI | No | Better visual feedback |
| Comprehensive Training | No | Complete guide included |
| Setup Automation | No | Automated setup script |

---

## Technical Architecture

```
Screen Capture (GStreamer) -> Center Crop -> Downscale
  -> YOLOv8 ONNX Inference -> NMS + Confidence Filter
  -> Team Color Detection (HSV body sampling)
  -> Target Scoring (distance * confidence * stickiness)
  -> Aim Smoothing (cubic ease-out + stealth jitter/overshoot)
  -> Anti-Recoil Compensation
  -> uinput Virtual Mouse Output
```

### Key Classes
1. `UMouse` - uinput virtual mouse (relative moves + clicks)
2. `BhopEngine` - uinput virtual keyboard for auto-jump
3. `AntiRecoil` - Recoil compensation with ramp, drift, and noise
4. `AdvancedToggle` - libevdev keyboard input, F-key handling
5. `StatusOverlay` - Secondary monitor overlay (thread-safe)
6. `ScreenCapture` - GStreamer X11/Wayland capture
7. `HeadDetector` - ONNX Runtime YOLOv8 inference
8. `TeamDetector` - HSV body-region color classification
9. `StealthEngine` - Jitter, overshoot, offsets, fire rate
10. `AimSmoother` - Cubic ease-out with dynamic factor
11. `CS2Aimbot` - Main loop tying everything together

---

## System Requirements

- **OS**: Linux (Ubuntu 20.04+, Arch Linux, EndeavourOS, Fedora, and derivatives)
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA (optional but recommended)
- **Storage**: 2GB for models and dependencies

---

## Troubleshooting

### Compilation Errors
```bash
# Missing dependencies
sudo apt install -y libopencv-dev libonnxruntime-dev libevdev-dev

# Wrong C++ standard
g++ -std=c++20 -O3 ...

# OpenCV pkg-config name (opencv4 vs opencv5)
# The setup script auto-detects this; if it fails, link specific modules:
g++ -std=c++20 -O3 cs2-cv-aimbot-improved.cpp \
    -lonnxruntime -lopencv_core -lopencv_imgproc -lopencv_highgui \
    -lopencv_dnn -lopencv_videoio -levdev -lpthread \
    -o cs2-aimbot-enhanced
```

### Permission Issues
```bash
# Fix input permissions
sudo usermod -aG video,input $USER
sudo chmod 666 /dev/uinput
# Reboot or re-login for group changes to take effect
```

### Model Loading
```bash
# Verify ONNX model
python3 -c "import onnx; print(onnx.load('cs2head.onnx').graph.input)"
```

### Performance Issues
- Lower `max_fps` in config.json
- Reduce `downscale_width/height` in config.json
- Enable GPU acceleration (`"use_gpu": true` in config.json)
- Use quantized models

---

## Legal and Safety

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

## License

This project is licensed under the MIT License - see the original README for details.

**Disclaimer**: The authors are not responsible for any damages, bans, or legal issues resulting from the use of this software. Use responsibly and in accordance with game terms of service.
