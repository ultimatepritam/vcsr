#!/bin/bash
# Setup script for Fast Downward and VAL under Linux/WSL.
# Run this inside WSL: bash /mnt/c/Expo/Research/PDDL/tools/setup_planning_tools.sh

set -e

echo "=== Installing build dependencies ==="
sudo apt-get update
sudo apt-get install -y cmake g++ make python3 git flex bison

# Fast Downward
FD_DIR="/opt/downward"
if [ ! -d "$FD_DIR" ]; then
    echo "=== Cloning Fast Downward ==="
    sudo git clone https://github.com/aibasel/downward.git "$FD_DIR"
    cd "$FD_DIR"
    echo "=== Building Fast Downward ==="
    sudo python3 build.py
    echo "Fast Downward installed at $FD_DIR"
else
    echo "Fast Downward already exists at $FD_DIR"
fi

# Verify Fast Downward
echo "=== Verifying Fast Downward ==="
python3 "$FD_DIR/fast-downward.py" --help | head -5

# VAL
VAL_DIR="/opt/VAL"
if [ ! -d "$VAL_DIR" ]; then
    echo "=== Cloning VAL ==="
    sudo git clone https://github.com/KCL-Planning/VAL.git "$VAL_DIR"
    cd "$VAL_DIR"
    echo "=== Building VAL ==="
    sudo mkdir -p build && cd build
    sudo cmake .. -DCMAKE_INSTALL_PREFIX="$VAL_DIR"
    sudo make -j$(nproc)
    sudo make install 2>/dev/null || true
    echo "VAL installed at $VAL_DIR"
else
    echo "VAL already exists at $VAL_DIR"
fi

# Verify VAL
echo "=== Verifying VAL ==="
if [ -f "$VAL_DIR/build/bin/Validate" ]; then
    echo "VAL binary found at $VAL_DIR/build/bin/Validate"
    export WSL_VAL_PATH="$VAL_DIR/build/bin/Validate"
elif [ -f "$VAL_DIR/validate" ]; then
    echo "VAL binary found at $VAL_DIR/validate"
else
    echo "WARNING: VAL binary not found. Check build output."
fi

echo ""
echo "=== Setup complete ==="
echo "Set these environment variables in your Python project:"
echo "  export FAST_DOWNWARD_PATH=$FD_DIR/fast-downward.py"
echo "  export VAL_PATH=$VAL_DIR/build/bin/Validate"
echo ""
echo "Or for WSL access from Windows:"
echo "  WSL_FAST_DOWNWARD_PATH=$FD_DIR/fast-downward.py"
echo "  WSL_VAL_PATH=$VAL_DIR/build/bin/Validate"
