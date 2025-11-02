#!/bin/bash

# CS2 CV Aimbot Enhanced Setup Script
# This script helps set up the enhanced CS2 CV aimbot

echo ""
echo "  ____ ____  _   _ _____      _     "
echo " / ___|  _ \| | | |_   _|   / \    "
echo "| |   | |_) | | | | | |    / _ \   "
echo "| |___|  _ <| |_| | | |   / ___ \  "
echo " \____|_| \\\\___/  |_|  /_/   \_\ "
echo "  Computer Vision Aimbot v2.0 Enhanced"
echo ""
echo "[!] Setting up CS2 CV Aimbot Enhanced..."
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "[!] Do not run this script as root. Run as regular user with sudo when needed."
    exit 1
fi

# Detect distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    DIST=$ID
else
    echo "[!] Cannot detect Linux distribution"
    exit 1
fi

echo "[+] Detected distribution: $OS"

# Function to install packages based on distribution
install_packages() {
    local packages="$1"
    
    case $DIST in
        ubuntu|debian)
            echo "[+] Installing packages for Debian/Ubuntu..."
            sudo apt update
            sudo apt install -y $packages
            ;;
        arch|manjaro)
            echo "[+] Installing packages for Arch Linux..."
            sudo pacman -S --noconfirm $packages
            ;;
        fedora)
            echo "[+] Installing packages for Fedora..."
            sudo dnf install -y $packages
            ;;
        *)
            echo "[!] Unsupported distribution: $DIST"
            echo "[!] Please install packages manually: $packages"
            exit 1
            ;;
    esac
}

# Install system dependencies
echo ""
echo "[+] Installing system dependencies..."

case $DIST in
    ubuntu|debian)
        DEB_PACKAGES="build-essential cmake libopencv-dev libonnxruntime-dev libevdev-dev libv4l-dev"
        install_packages "$DEB_PACKAGES"
        ;;
    arch|manjaro)
        ARCH_PACKAGES="base-devel cmake opencv onnxruntime libevdev v4l-utils"
        install_packages "$ARCH_PACKAGES"
        ;;
    fedora)
        FEDORA_PACKAGES="gcc-c++ cmake opencv-devel onnxruntime-devel libevdev-devel v4l-utils"
        install_packages "$FEDORA_PACKAGES"
        ;;
esac

# Check for NVIDIA CUDA
echo ""
echo "[+] Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "[+] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader
    
    # Install CUDA if not present
    if ! command -v nvcc &> /dev/null; then
        echo "[!] CUDA toolkit not found. Installing..."
        case $DIST in
            ubuntu|debian)
                sudo apt install -y nvidia-cuda-toolkit
                ;;
            arch|manjaro)
                sudo pacman -S --noconfirm cuda
                ;;
            fedora)
                sudo dnf install -y cuda-toolkit
                ;;
        esac
    fi
else
    echo "[!] No NVIDIA GPU detected. CPU inference will be used."
fi

# Set up Python environment
echo ""
echo "[+] Setting up Python environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "[!] Python 3 not found. Installing..."
    case $DIST in
        ubuntu|debian)
            sudo apt install -y python3 python3-pip
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm python python-pip
            ;;
        fedora)
            sudo dnf install -y python3 python3-pip
            ;;
    esac
fi

# Create virtual environment
if [ ! -d "cs2_training_env" ]; then
    echo "[+] Creating Python virtual environment..."
    python3 -m venv cs2_training_env
fi

# Activate virtual environment
source cs2_training_env/bin/activate

# Install Python packages
echo "[+] Installing Python packages..."
pip install --upgrade pip
pip install ultralytics opencv-python numpy pillow matplotlib tensorboard roboflow

# Set up permissions
echo ""
echo "[+] Setting up system permissions..."
sudo usermod -aG video,input $USER
sudo modprobe uinput

# Create udev rule for uinput
echo "[+] Creating udev rules..."
sudo tee /etc/udev/rules.d/99-uinput.rules > /dev/null <<EOF
KERNEL=="uinput", MODE="0666", GROUP="input"
EOF

sudo udevadm control --reload-rules

# Compile the aimbot
echo ""
echo "[+] Compiling the enhanced aimbot..."
if [ -f "cs2-cv-aimbot-improved.cpp" ]; then
    g++ -std=c++20 -O3 cs2-cv-aimbot-improved.cpp \
        -lonnxruntime \
        -lopencv_core \
        -lopencv_imgproc \
        -lopencv_highgui \
        -levdev \
        -o cs2-aimbot-enhanced
    
    if [ $? -eq 0 ]; then
        echo "[✓] Compilation successful!"
        echo "[+] Executable: ./cs2-aimbot-enhanced"
    else
        echo "[!] Compilation failed!"
        echo "[!] Please check the error messages above."
        exit 1
    fi
else
    echo "[!] Source file not found: cs2-cv-aimbot-improved.cpp"
    exit 1
fi

# Create directories
echo ""
echo "[+] Creating necessary directories..."
mkdir -p datasets models screenshots

# Download sample model (optional)
echo ""
echo "[+] Would you like to download a sample model? (y/n)"
read -r download_model

if [ "$download_model" = "y" ] || [ "$download_model" = "Y" ]; then
    echo "[!] Sample model download not implemented yet."
    echo "[!] Please train your own model using the training guide."
fi

# Create configuration file
echo ""
echo "[+] Creating configuration file..."
cat > config.json <<EOF
{
    "aimbot": {
        "fov": 120,
        "smooth": 8.0,
        "trigger_threshold": 0.22,
        "max_fps": 60,
        "confidence_threshold": 0.35
    },
    "controls": {
        "team_toggle": "F7",
        "aimbot_toggle": "F8",
        "debug_toggle": "F9",
        "status_key": "F10"
    },
    "display": {
        "debug_overlay": false,
        "show_fps": true,
        "show_team_detection": true
    }
}
EOF

# Create training dataset structure
echo ""
echo "[+] Creating training dataset structure..."
mkdir -p cs2_dataset/{images/{train,val,test},labels/{train,val,test}}

# Final instructions
echo ""
echo "==============================================="
echo "Setup completed successfully!"
echo "==============================================="
echo ""
echo "Next steps:"
echo "1. Read TRAINING_GUIDE.md for detailed instructions"
echo "2. Collect CS2 screenshots in cs2_dataset/images/"
echo "3. Label the data using CVAT or makesense.ai"
echo "4. Train your model using the provided commands"
echo "5. Place the trained model as 'cs2head.onnx' in this directory"
echo "6. Run: sudo ./cs2-aimbot-enhanced"
echo ""
echo "Controls:"
echo "  F7 - Toggle target team (T/CT)"
echo "  F8 - Toggle aimbot on/off"
echo "  F9 - Toggle debug overlay"
echo "  F10 - Show status"
echo ""
echo "IMPORTANT:"
echo "- This tool is for educational purposes only"
echo "- Use at your own risk - may result in game bans"
echo "- Always test in offline mode first"
echo "- Never use on your main gaming account"
echo ""
echo "For more information, read the TRAINING_GUIDE.md"
echo "==============================================="