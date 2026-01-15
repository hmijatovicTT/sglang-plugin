# SGLang TT-Metal Plugin

This plugin adds TT-Metal device support to SGLang, allowing you to run language model inference on TT-Metal hardware accelerators.

## Installation

⚠️ **CRITICAL**: For CPU-only setups, you must install CPU PyTorch **BEFORE** SGLang. See [INSTALLATION.md](INSTALLATION.md) for detailed instructions.

### Quick Installation

1. **Uninstall conflicting packages:**
   ```bash
   pip uninstall -y torch torchvision torchaudio sglang sgl-kernel
   pip cache purge
   ```

2. **Install CPU PyTorch FIRST:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Set environment variable:**
   ```bash
   export SGLANG_USE_CPU_ENGINE=1
   ```

4. **Install SGLang:**
   ```bash
   pip install sglang
   ```

5. **Install the plugin:**
   ```bash
   cd sglang-plugin
   pip install -e .
   ```

For detailed instructions and troubleshooting, see [INSTALLATION.md](INSTALLATION.md).

## Usage

Simply import the plugin before starting your SGLang server:

```python
import sglang_tt_plugin  # This registers TT-Metal models automatically

# Now start SGLang normally
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
```

Or use environment variable:

```bash
export SGLANG_TT_PLUGIN=1
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
```

## Features

- **Automatic Model Registration**: TT-Metal models are automatically registered when the plugin is imported
- **Drop-in Replacement**: Works with existing SGLang configurations
- **TT-Metal Integration**: Leverages TT-Metal's optimized inference capabilities
- **Clean Separation**: No modifications to SGLang core code required

## Requirements

- **CPU PyTorch** (must be installed FIRST - see installation instructions)
- SGLang >= 0.3.0
- TT-Metal (ttnn)
- Python >= 3.8

**Note**: This plugin requires CPU-only PyTorch. GPU PyTorch will cause `sgl_kernel`/`common_ops` errors.

## Architecture

The plugin provides:
- `TTLlamaForCausalLM`: TT-Metal optimized Llama implementation
- Utility functions for TT-Metal device management
- Automatic registration with SGLang's model registry

## Supported Models

Currently supported:
- LlamaForCausalLM (all Llama variants)

## Development

To develop the plugin:

1. Clone this repository
2. Install in development mode: `pip install -e .`
3. Make changes to the plugin code
4. Test with SGLang

## License

Apache 2.0