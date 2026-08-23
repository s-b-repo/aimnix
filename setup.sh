#!/bin/bash
set -e

echo ""
echo "  ____ ____  _   _ _____      _     "
echo " / ___|  _ \| | | |_   _|   / \    "
echo "| |   | |_) | | | | | |    / _ \   "
echo "| |___|  _ <| |_| | | |   / ___ \  "
echo " \____|_| \\\\___/  |_|  /_/   \_\ "
echo "  Computer Vision Aimbot v3.0 Setup"
echo ""

if [ "$EUID" -eq 0 ]; then
    echo "[!] Do not run this script as root. Run as regular user."
    exit 1
fi

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    DIST=$ID
else
    echo "[!] Cannot detect Linux distribution"
    exit 1
fi

echo "[+] Detected: $OS ($DIST)"

install_packages() {
    local packages="$1"

    case $DIST in
        ubuntu|debian|kali|pop|linuxmint|elementary|zorin)
            echo "[+] Installing packages (apt)..."
            sudo apt update
            sudo apt install -y $packages
            ;;
        arch|manjaro|endeavouros|garuda)
            echo "[+] Installing packages (pacman)..."
            sudo pacman -S --noconfirm $packages
            ;;
        fedora|nobara)
            echo "[+] Installing packages (dnf)..."
            sudo dnf install -y $packages
            ;;
        opensuse*|suse*)
            echo "[+] Installing packages (zypper)..."
            sudo zypper install -y $packages
            ;;
        *)
            echo "[!] Unsupported distribution: $DIST"
            echo "[!] Please install manually: $packages"
            exit 1
            ;;
    esac
}

verify_command() {
    if command -v "$1" &> /dev/null; then
        echo "[+] Verified: $1"
        return 0
    else
        echo "[!] Missing: $1"
        return 1
    fi
}

echo ""
echo "[+] Installing system dependencies..."

case $DIST in
    ubuntu|debian|kali|pop|linuxmint|elementary|zorin)
        install_packages "build-essential cmake pkg-config libopencv-dev libevdev-dev libgstreamer1.0-dev gstreamer1.0-plugins-good"
        # onnxruntime may not be in repos; handle separately
        if ! dpkg -l libonnxruntime-dev &> /dev/null 2>&1; then
            echo "[+] Installing libonnxruntime-dev..."
            sudo apt install -y libonnxruntime-dev 2>/dev/null || {
                echo "[!] libonnxruntime-dev not in repos. Install manually:"
                echo "    https://github.com/microsoft/onnxruntime/releases"
            }
        fi
        ;;
    arch|manjaro|endeavouros|garuda)
        install_packages "base-devel cmake pkgconf opencv libevdev gst-plugins-good"
        if ! pacman -Qi onnxruntime &> /dev/null 2>&1; then
            echo "[+] Installing onnxruntime..."
            sudo pacman -S --noconfirm onnxruntime-cuda 2>/dev/null || {
                echo "[!] Trying onnxruntime-cpu..."
                sudo pacman -S --noconfirm onnxruntime-cpu 2>/dev/null || true
            }
        fi
        # Download onnxruntime C++ headers if not present
        if [ ! -f "third_party/onnxruntime/include/onnxruntime_cxx_api.h" ]; then
            echo "[+] Downloading onnxruntime C++ headers..."
            ORT_VERSION="1.28.0"
            wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" -O /tmp/ort-headers.tgz 2>/dev/null && {
                mkdir -p third_party/onnxruntime/include third_party/onnxruntime/lib
                tar xzf /tmp/ort-headers.tgz -C /tmp/
                cp /tmp/onnxruntime-linux-x64-${ORT_VERSION}/include/*.h third_party/onnxruntime/include/
                cp -r /tmp/onnxruntime-linux-x64-${ORT_VERSION}/include/core third_party/onnxruntime/include/
                cp /tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so* third_party/onnxruntime/lib/ 2>/dev/null || true
                echo "[+] onnxruntime headers installed in third_party/onnxruntime/"
            } || echo "[!] Could not download headers; compilation may fail"
        fi
        ;;
    fedora|nobara)
        install_packages "gcc-c++ cmake pkgconf-pkg-config opencv-devel libevdev-devel gstreamer1-plugins-good"
        sudo dnf install -y onnxruntime-devel 2>/dev/null || {
            echo "[!] onnxruntime-devel not in repos. Install manually."
        }
        ;;
esac

echo ""
echo "[+] Verifying core tools..."
MISSING=0
for cmd in g++ cmake pkg-config; do
    verify_command "$cmd" || MISSING=1
done
if [ "$MISSING" -eq 1 ]; then
    echo "[!] Some tools are missing. Fix before continuing."
    exit 1
fi

echo ""
echo "[+] Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "[+] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

    if ! command -v nvcc &> /dev/null; then
        echo "[+] CUDA toolkit not found. Installing..."
        case $DIST in
            ubuntu|debian|kali|pop|linuxmint)
                sudo apt install -y nvidia-cuda-toolkit 2>/dev/null || echo "[!] CUDA install failed; install manually"
                ;;
            arch|manjaro|endeavouros|garuda)
                sudo pacman -S --noconfirm cuda 2>/dev/null || echo "[!] CUDA install failed; install manually"
                ;;
            fedora|nobara)
                sudo dnf install -y cuda-toolkit 2>/dev/null || echo "[!] CUDA install failed; install manually"
                ;;
        esac
    else
        echo "[+] CUDA toolkit already installed"
    fi
else
    echo "[!] No NVIDIA GPU detected. CPU inference will be used."
fi

echo ""
echo "[+] Setting up Python environment..."

if ! command -v python3 &> /dev/null; then
    echo "[+] Python 3 not found. Installing..."
    case $DIST in
        ubuntu|debian|kali|pop|linuxmint|elementary|zorin)
            sudo apt install -y python3 python3-pip python3-venv
            ;;
        arch|manjaro|endeavouros|garuda)
            sudo pacman -S --noconfirm python python-pip
            ;;
        fedora|nobara)
            sudo dnf install -y python3 python3-pip
            ;;
    esac
fi

if [ ! -d "cs2_training_env" ]; then
    echo "[+] Creating Python virtual environment..."
    python3 -m venv cs2_training_env
fi

source cs2_training_env/bin/activate

echo "[+] Installing Python packages..."
pip install --upgrade pip --quiet
# Install CPU-only torch to avoid downloading ~1.5GB of NVIDIA CUDA packages
# For GPU training, replace with: pip install torch --quiet
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet 2>/dev/null || pip install torch --quiet
pip install ultralytics opencv-python-headless numpy pillow mss keyboard --quiet
echo "[+] Python packages installed"

echo ""
echo "[+] Setting up system permissions..."
sudo usermod -aG video,input "$USER" 2>/dev/null || true
sudo modprobe uinput 2>/dev/null || true

if [ ! -f /etc/udev/rules.d/99-uinput.rules ]; then
    echo "[+] Creating udev rules for uinput..."
    sudo tee /etc/udev/rules.d/99-uinput.rules > /dev/null <<EOF
KERNEL=="uinput", MODE="0666", GROUP="input"
EOF
    sudo udevadm control --reload-rules
else
    echo "[+] uinput rules already exist"
fi

echo ""
echo "[+] Compiling..."
if [ -f "cs2-cv-aimbot-improved.cpp" ]; then
    # Detect OpenCV pkg-config name (opencv4 vs opencv5)
    if pkg-config --exists opencv4 2>/dev/null; then
        OPENCV_CFLAGS=$(pkg-config --cflags opencv4)
        OPENCV_LIBS=$(pkg-config --libs opencv4)
    elif pkg-config --exists opencv5 2>/dev/null; then
        OPENCV_CFLAGS=$(pkg-config --cflags opencv5)
        OPENCV_LIBS=$(pkg-config --libs opencv5)
    elif pkg-config --exists opencv 2>/dev/null; then
        OPENCV_CFLAGS=$(pkg-config --cflags opencv)
        OPENCV_LIBS=$(pkg-config --libs opencv)
    else
        echo "[!] OpenCV not found via pkg-config. Trying direct linking..."
        OPENCV_CFLAGS=""
        OPENCV_LIBS="-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_dnn -lopencv_videoio"
    fi

    # Detect libevdev pkg-config
    if pkg-config --exists libevdev 2>/dev/null; then
        EVDEV_CFLAGS=$(pkg-config --cflags libevdev)
    else
        EVDEV_CFLAGS=""
    fi

    # Check if onnxruntime headers are available
    ONNX_INC=""
    ONNX_LIB=""
    if [ -d "third_party/onnxruntime/include" ]; then
        ONNX_INC="-I third_party/onnxruntime/include"
        ONNX_LIB="-L third_party/onnxruntime/lib -Wl,-rpath,third_party/onnxruntime/lib"
        echo "[+] Using bundled onnxruntime in third_party/"
    elif pkg-config --exists onnxruntime 2>/dev/null; then
        ONNX_INC=$(pkg-config --cflags onnxruntime)
        ONNX_LIB=$(pkg-config --libs onnxruntime)
    fi

    # Try full opencv libs first, fall back to specific modules if VTK/HDF5 deps are missing
    if [ -n "$OPENCV_LIBS" ]; then
        g++ -std=c++20 -O3 cs2-cv-aimbot-improved.cpp \
            $OPENCV_CFLAGS $OPENCV_LIBS \
            $EVDEV_CFLAGS $ONNX_INC $ONNX_LIB \
            -lonnxruntime \
            -levdev \
            -lpthread \
            -o cs2-aimbot-enhanced 2>&1
    fi

    # If that failed (or OPENCV_LIBS was empty), try with specific modules only
    if [ ! -f "cs2-aimbot-enhanced" ] || [ $? -ne 0 ]; then
        echo "[+] Retrying with specific OpenCV modules (avoids VTK/HDF5 deps)..."
        g++ -std=c++20 -O3 cs2-cv-aimbot-improved.cpp \
            $OPENCV_CFLAGS $EVDEV_CFLAGS $ONNX_INC $ONNX_LIB \
            -lonnxruntime \
            -lopencv_core -lopencv_imgproc -lopencv_highgui \
            -lopencv_dnn -lopencv_videoio \
            -levdev \
            -lpthread \
            -Wl,-rpath,third_party/onnxruntime/lib \
            -o cs2-aimbot-enhanced 2>&1
    fi

    if [ $? -eq 0 ]; then
        echo "[+] Compilation successful: ./cs2-aimbot-enhanced"
    else
        echo "[!] Compilation failed. Check errors above."
        echo "[!] Make sure onnxruntime headers and library are available:"
        echo "    - Install system package: onnxruntime-cuda (Arch) or libonnxruntime-dev (Ubuntu)"
        echo "    - Or download from https://github.com/microsoft/onnxruntime/releases"
        echo "      and place in third_party/onnxruntime/"
        exit 1
    fi
else
    echo "[!] Source file not found: cs2-cv-aimbot-improved.cpp"
    exit 1
fi

echo ""
echo "[+] Creating directories..."
mkdir -p cs2_dataset/{images/{train,val,test},labels/{train,val,test}}

if [ ! -f "config.json" ]; then
    echo "[+] Creating default config.json..."
    cat > config.json <<'CONFIGEOF'
{
    "aimbot": {
        "fov": 120,
        "smooth": 8.0,
        "trigger_threshold": 0.22,
        "max_fps": 60,
        "confidence_threshold": 0.35,
        "downscale_width": 1280,
        "downscale_height": 720,
        "target_team": "T",
        "auto_shoot": true
    },
    "controls": {
        "team_toggle": "F7",
        "aimbot_toggle": "F8",
        "debug_toggle": "F9",
        "status_key": "F10"
    },
    "performance": {
        "use_gpu": true,
        "gpu_device": 0,
        "cpu_threads": 4
    },
    "detection": {
        "model_path": "cs2head.onnx",
        "input_size": 640,
        "nms_threshold": 0.4,
        "color_threshold": 50
    },
    "safety": {
        "max_aim_speed": 100,
        "reaction_delay_min": 50,
        "reaction_delay_max": 200,
        "miss_probability": 0.05
    }
}
CONFIGEOF
else
    echo "[+] config.json already exists, skipping"
fi

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Collect data:  python3 collect_data.py --rate 5"
echo "  2. Label data:    Use CVAT or makesense.ai"
echo "  3. Train model:   python3 train_model.py --all"
echo "  4. Run:           sudo ./cs2-aimbot-enhanced"
echo ""
echo "Controls: F7=team  F8=aimbot  F9=debug  F10=status"
echo "========================================"
