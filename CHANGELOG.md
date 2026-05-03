# Changelog

## v3.0

### Aim Engine
- Replaced linear smoothing with cubic ease-out curve and dynamic smooth factor
- Added Assist mode (smooth tracking) and Lock mode (snap while holding key)
- Added dead zone to suppress micro-adjustments when crosshair is near target
- Added weighted target scoring combining distance, confidence, and target stickiness
- Replaced hardcoded trigger (box overlap ratio) with crosshair-in-bbox check
- All config values loaded from config.json at runtime instead of compile-time constants

### Stealth System (new)
- Gaussian micro-jitter on aim movement path
- Random overshoot with sinusoidal correction phase
- Per-frame aim speed variance
- Per-target random aim offset within bounding box
- Target switch penalty discourages rapid target flicking
- Randomized reaction delay on first acquiring a target
- Configurable miss probability
- Fire rate limiter with minimum interval between shots
- Process name masking from pool of 27 common Linux process names via prctl

### Anti-Recoil (new)
- Automatic downward compensation during sustained fire
- Strength ramps up over burst duration
- Randomized horizontal drift generated per spray
- Gaussian noise on both vertical and horizontal compensation
- Periodic micro-pauses in compensation for human-like variance

### Bunny Hop (new)
- Auto-jump via uinput virtual keyboard when holding space
- Toggleable with F6
- Randomized tap duration (10-24ms)

### Team Detection (rewritten)
- Moved color sampling from head region to body region below head
- Pixel-level HSV classification instead of mean color
- Added saturation and value thresholds to filter noise
- T-side hue range: 0-25 and 155-180 (reds, oranges, yellows)
- CT-side hue range: 95-135 (blues)
- Requires 25%+ pixel ratio and 1.5x dominance to classify

### Status Overlay (new)
- Renders on secondary monitor via OpenCV window
- Shows control state changes (team, mode, aimbot, panic)
- Thread-safe rendering with mutex-protected state
- Auto-clears after configurable duration

### Controls (expanded)
- F5: Toggle aim mode (Assist / Lock)
- F6: Toggle bunny hop
- F7: Toggle target team (T / CT)
- F8: Toggle aimbot on/off
- F9: Toggle debug overlay
- F10: Show all current states (5 second display)
- F11: Hot-reload config.json
- F12: Panic kill-switch (disables everything, destroys windows)
- Left Alt: Hold for aim-lock in Lock mode

### Screen Capture (improved)
- Added center-screen cropping for faster inference
- Added Wayland support via PipeWire GStreamer source
- Deadline-based frame timing (sleep = target - elapsed)

### Config System (rewritten)
- All settings loaded from config.json with section-based JSON parser
- Input validation and clamping to prevent division-by-zero and UB
- Reaction delay min/max swap if inverted
- Hot-reload via F11 updates aim, safety, and recoil parameters

### Bug Fixes
- Replaced thread-unsafe rand() with per-class std::mt19937 + distributions
- Fixed StatusOverlay show_until data race with mutex protection
- Fixed StatusOverlay Clock typedef ordering (declared before use)
- Added model file existence check before ONNX session creation
- Clamped negative detection box coordinates and zero-size boxes
- Added bounds checking on detection boxes against image dimensions
- Removed unused mouse1_held atomic variable
- Fixed BhopEngine ioctl error handling (cleanup on failure)
- Fixed F6 bhop toggle was no-op (overlay text only, never called toggle)
- Fixed BhopEngine lazy init: toggling on after disabled-start now opens uinput
- Fixed unchecked write() calls in UMouse and BhopEngine (emit() wrapper)
- Fixed ScreenCapture scale_x/y division-by-zero when GStreamer returns 0 dimensions
- Wired config.json `use_gpu` to skip CUDA when false
- Wired config.json `debug_overlay` as initial debug state
- Hot-reload now propagates stealth/smoother fields via reload() methods

### Data Collection (collect_data.py)
- Thread-safe capture with threading.Event and Lock
- Edge-triggered F9 key detection (no repeated toggles)
- Duplicate frame detection via perceptual hash
- Auto val/train split during capture
- Configurable JPEG quality and capture region
- Added cv2.imwrite return value checking
- Added mss.grab() exception handling
- Added disk space monitoring (warns at < 500MB)

### Training (train_model.py)
- Auto train/val split with configurable ratio
- Resume from last checkpoint
- Support for all image formats
- Finds best/last weights by scanning runs/detect/
- Added is_dir() filter to prevent crashes on non-directory entries
- Added shutil.move error handling in auto_split
- Added shutil.copy error handling in export_model

### Setup (setup.sh)
- Supports Kali, Pop!_OS, Linux Mint, Elementary, Zorin, EndeavourOS, Garuda, Nobara
- Verifies core tools after installation
- Better onnxruntime availability handling
- Creates dataset directory structure
- Sets up Python virtual environment with all dependencies

---

## v2.1

### Critical Bug Fixes (C++)
- Fixed uinput API calls (replaced fake functions with correct ioctl calls)
- Fixed YOLOv8 output tensor parsing (transposed layout)
- Fixed screen capture (GStreamer instead of webcam)
- Fixed ONNX Runtime API (managed name pointers)
- Fixed CUDA provider initialization
- Fixed compile command (pkg-config, missing libs)

### Runtime Bug Fixes (C++)
- Added NMS (Non-Maximum Suppression) for duplicate box filtering
- Fixed AdvancedToggle destructor race condition
- Replaced fragile keyboard name matching with capability detection
- Added graceful shutdown via signal handler
- Fixed member initialization order for Ort::Session/Env
- Fixed team detection out-of-bounds reads
- Fixed mouse movement scaling for downscaled frames
- Fixed FPS calculation divide-by-zero

### Python Bug Fixes
- Fixed --duration being ignored in collect_data.py
- Fixed F9 toggle debounce
- Fixed leading newline in generated data.yaml
- Fixed invalid export parameter in train_model.py

### Shell Script Bug Fixes
- Fixed config.json overwrite with incomplete version
- Added GStreamer dependencies
- Added set -e for fail-fast
- Fixed compile command

## v2.0
- YOLOv8 ONNX inference
- uinput virtual mouse
- Hardcoded configuration constants
- Linear aim smoothing
