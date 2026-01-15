# SGLang TT-Plugin Installation Guide

## Critical Installation Order

⚠️ **IMPORTANT**: The installation order is **critical** for CPU-only setups. If you install packages in the wrong order, `sgl_kernel` and `common_ops` may be built for GPU instead of CPU, causing runtime errors like:
- `ImportError: Could not load any common_ops library!`
- `ImportError: libnvrtc.so.12: cannot open shared object file`

## Installation Steps

### Step 1: Uninstall Conflicting Packages

First, remove any existing PyTorch and SGLang installations:

```bash
pip uninstall -y torch torchvision torchaudio sglang sgl-kernel
pip cache purge  # Clear pip cache to avoid cached GPU versions
```

### Step 2: Install CPU PyTorch FIRST ⚠️

**This is the most critical step.** You MUST install CPU PyTorch before SGLang:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Why this order matters:**
- `pip install sglang` does not guarantee correct `sgl_kernel/common_ops` for CPU
- If GPU PyTorch is installed first, `sgl_kernel` will be the GPU version
- Installing CPU PyTorch first ensures `sgl_kernel` builds for CPU

### Step 3: Verify CPU PyTorch Installation

Verify that CPU PyTorch is installed correctly:

```bash
python3 -c "import torch; assert torch.version.cuda is None, 'GPU PyTorch detected!'; print(f'✓ CPU PyTorch {torch.__version__} installed correctly')"
```

If you see "GPU PyTorch detected!", go back to Step 1 and uninstall PyTorch again.

### Step 4: Set Environment Variables

Set the CPU engine environment variable:

```bash
export SGLANG_USE_CPU_ENGINE=1
```

To make this permanent, add it to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export SGLANG_USE_CPU_ENGINE=1' >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Install SGLang

Now install SGLang:

```bash
pip install sglang
```

### Step 6: Ensure Correct sgl-kernel Version (Optional but Recommended)

Reinstall `sgl-kernel` to ensure it's built for CPU:

```bash
pip uninstall -y sgl-kernel
pip install sgl-kernel
```

### Step 7: Install the Plugin

Finally, install the TT-Plugin:

```bash
cd sglang-plugin
pip install -e .
```

## Verification

After installation, verify everything works:

```bash
python3 -c "
import os
os.environ['SGLANG_USE_CPU_ENGINE'] = '1'

import torch
assert torch.version.cuda is None, 'GPU PyTorch detected!'
print(f'✓ PyTorch: {torch.__version__} (CPU)')

import sglang
print(f'✓ SGLang: installed')

import sglang_tt_plugin
print(f'✓ TT-Plugin: {sglang_tt_plugin.__version__}')
print('✓ All packages installed correctly!')
"
```

## Usage

After installation, simply import the plugin before starting your SGLang server:

```python
import sglang_tt_plugin  # This monkey patches LlamaForCausalLM -> TTLlamaForCausalLM

# Now start SGLang normally - it will use TT-Metal models
python -m sglang.launch_server --model <MODEL_ID> --device cpu
```

## Troubleshooting

### Error: "Could not load any common_ops library!"

This means `sgl_kernel` or `common_ops` is the wrong version (GPU instead of CPU).

**Solution:**
1. Uninstall everything: `pip uninstall -y torch torchvision torchaudio sglang sgl-kernel`
2. Clear pip cache: `pip cache purge`
3. Follow the installation steps above in order
4. Make sure CPU PyTorch is installed BEFORE SGLang

### Error: "libnuma.so.1: cannot open shared object file"

Missing system library required by `sgl_kernel`.

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install numactl

# CentOS/RHEL
sudo yum install numactl
```

### Error: "libnvrtc.so.12: cannot open shared object file"

This means `sgl_kernel` is trying to load GPU libraries on a CPU-only system.

**Solution:**
1. Uninstall and reinstall following the correct order
2. Ensure CPU PyTorch is installed first
3. Reinstall `sgl-kernel` after SGLang

### Error: "undefined symbol" or ABI mismatch

PyTorch and `sgl_kernel` versions are incompatible.

**Solution:**
1. Uninstall both: `pip uninstall -y torch sgl-kernel`
2. Install CPU PyTorch first
3. Install SGLang (which will pull compatible sgl-kernel)
4. If issues persist, reinstall sgl-kernel: `pip uninstall -y sgl-kernel && pip install sgl-kernel`

### GPU PyTorch Detected After Installation

If verification shows GPU PyTorch is installed:

**Solution:**
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -c "import torch; assert torch.version.cuda is None; print('OK')"
```

## Summary

The key points:
1. ✅ Install CPU PyTorch **FIRST**
2. ✅ Set `SGLANG_USE_CPU_ENGINE=1` environment variable
3. ✅ Install SGLang after PyTorch
4. ✅ Reinstall `sgl-kernel` to ensure CPU version
5. ✅ Install plugin last
6. ✅ Import plugin before SGLang: `import sglang_tt_plugin`

If you encounter `sgl_kernel` or `common_ops` errors, it usually means the installation order was wrong or the environment wasn't clean. Start over with a clean uninstall and follow the steps in order.
