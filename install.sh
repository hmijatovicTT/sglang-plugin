#!/bin/bash
# Installation script for SGLang TT-Plugin
# This ensures correct installation order to avoid sgl_kernel/common_ops CPU/GPU mismatches

set -e  # Exit on error

echo "=========================================="
echo "SGLang TT-Plugin Installation Script"
echo "=========================================="
echo ""
echo "This script ensures proper installation order:"
echo "1. CPU PyTorch (to ensure sgl_kernel builds for CPU)"
echo "2. SGLang"
echo "3. TT-Plugin"
echo ""

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: Not in a virtual environment. Consider using one."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Install CPU PyTorch
echo "Step 1: Installing CPU PyTorch..."
echo "This is CRITICAL - installing CPU PyTorch first ensures sgl_kernel builds for CPU"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch is CPU version
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; assert torch.version.cuda is None, 'ERROR: PyTorch has CUDA support! Install CPU version.'; print('✓ PyTorch CPU version confirmed')"

# Step 2: Install system dependencies (numactl for sgl_kernel)
echo ""
echo "Step 2: Checking system dependencies..."
if ! command -v numactl &> /dev/null; then
    echo "WARNING: numactl not found. sgl_kernel may fail with 'libnuma.so.1' error."
    echo "Install with: sudo apt-get install numactl (Ubuntu/Debian)"
    echo "or: sudo yum install numactl (RHEL/CentOS)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ numactl found"
fi

# Step 3: Install SGLang
echo ""
echo "Step 3: Installing SGLang..."
# Set environment variable before installation
export SGLANG_USE_CPU_ENGINE=1
pip install sglang

# Verify sglang installation
echo ""
echo "Verifying SGLang installation..."
python -c "import sglang; print(f'✓ SGLang {sglang.__version__} installed')" || {
    echo "ERROR: SGLang installation failed"
    exit 1
}

# Step 4: Install the plugin
echo ""
echo "Step 4: Installing TT-Plugin..."
pip install -e .

# Step 5: Verify installation
echo ""
echo "Step 5: Verifying plugin installation..."
python -c "
import sglang_tt_plugin
print('✓ TT-Plugin imported successfully')
from sglang_tt_plugin.models.tt_llama import TTLlamaForCausalLM
print('✓ TTLlamaForCausalLM imported successfully')
" || {
    echo "ERROR: Plugin installation verification failed"
    exit 1
}

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Before running SGLang, set:"
echo "  export SGLANG_USE_CPU_ENGINE=1"
echo ""
echo "Then launch SGLang with:"
echo "  python -m sglang.launch_server --model <MODEL_PATH> --device cpu"
echo ""
echo "Or use the provided script:"
echo "  sglang-tt-server --model-path <MODEL_PATH>"
echo ""
