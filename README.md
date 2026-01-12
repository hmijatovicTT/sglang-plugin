# SGLang TT-Metal Plugin

This plugin adds TT-Metal device support to SGLang, allowing you to run language model inference on TT-Metal hardware accelerators.

## Installation

```bash
cd sglang_tt_plugin
pip install -e .
```

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

- SGLang >= 0.3.0
- TT-Metal (ttnn)
- PyTorch
- Python >= 3.8

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