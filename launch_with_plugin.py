"""
Launch SGLang server with TT-Metal plugin monkey-patched in
Uses SGLang from source code instead of installed version
"""
import sys
import os

# Add sglang source code to Python path FIRST
SGLANG_SOURCE_PATH = "/localdev/hmijatovic/sglang/python"
if SGLANG_SOURCE_PATH not in sys.path:
    sys.path.insert(0, SGLANG_SOURCE_PATH)

# Add tt-metal to path
sys.path.insert(0, '/localdev/hmijatovic/tt-metal')
# Add current directory to path for plugin imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import plugin models BEFORE SGLang loads
from sglang_tt_plugin.models.tt_llama import TTLlamaForCausalLM
from sglang_tt_plugin.utils.tt_utils import open_mesh_device, close_mesh_device

# Now import SGLang (from globally installed version)
import sglang.srt.models

# Monkey-patch: inject plugin model into SGLang's models namespace
sglang.srt.models.TTLlamaForCausalLM = TTLlamaForCausalLM
sys.modules['sglang.srt.models.tt_llama'] = sys.modules['sglang_tt_plugin.models.tt_llama']

print("=" * 80)
print("TT-Metal Plugin Loaded")
print(f"SGLang location: {sglang.__file__}")
print(f"TTLlamaForCausalLM from: {TTLlamaForCausalLM.__module__}")
print("=" * 80)

# Launch SGLang server
if __name__ == "__main__":
    from sglang.launch_server import main
    main()