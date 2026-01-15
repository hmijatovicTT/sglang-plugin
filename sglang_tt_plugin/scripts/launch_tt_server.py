#!/usr/bin/env python3
"""
Launch SGLang server with TT-Metal plugin support.
Based on the working command structure.
"""

import os
import sys
import subprocess
import argparse

# Add sglang source code to Python path
SGLANG_SOURCE_PATH = "/localdev/hmijatovic/sglang/python"
if SGLANG_SOURCE_PATH not in sys.path:
    sys.path.insert(0, SGLANG_SOURCE_PATH)

def setup_tt_environment():
    """Setup TT-Metal environment variables similar to your working command."""
    # TT-Metal environment
    os.environ["VLLM_DEVICE_TYPE"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
    os.environ["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libnuma.so.1"
    os.environ["TRITON_CPU_ONLY"] = "1"
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ["SGLANG_TT_PLUGIN"] = "1"  # Enable auto-registration
    
    # Import TT plugin to monkey-patch SGLang (this will be done after SGLang loads)
    try:
        # Just test that the plugin is available, don't actually import it yet
        import importlib.util
        spec = importlib.util.find_spec("sglang_tt_plugin")
        if spec is None:
            raise ImportError("Plugin not found")
        print("[TT-Plugin] Plugin is available and ready for monkey-patching")
    except ImportError as e:
        print(f"[TT-Plugin] TT plugin not available: {e}")
        sys.exit(1)

def main():
    """Main entry point for TT server launch."""
    parser = argparse.ArgumentParser(description="Launch SGLang server with TT support")
    parser.add_argument("--model-path", required=True, help="Model path")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=30000, help="Port number")
    parser.add_argument("--page-size", type=int, default=128, help="Page size")
    parser.add_argument("--max-running-requests", type=int, default=1, help="Max running requests")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    parser.add_argument("--disable-overlap-schedule", action="store_true", help="Disable overlap schedule")
    
    args = parser.parse_args()
    
    # Setup TT environment
    setup_tt_environment()
    
    # Build SGLang command
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model_path,
        "--host", args.host,
        "--port", str(args.port),
        "--page-size", str(args.page_size),
        "--max-running-requests", str(args.max_running_requests),
        "--log-level", args.log_level,
    ]
    
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if args.disable_overlap_schedule:
        cmd.append("--disable-overlap-schedule")
    
    print(f"Launching SGLang TT server with command: {' '.join(cmd)}")
    print("Looking for 'Overwriting LlamaForCausalLM with TTLlamaForCausalLM' message...")
    
    # Build command that loads plugin AFTER SGLang starts
    python_cmd = [
        sys.executable, 
        "-c", 
        f"""
import os
import sys

# Import SGLang first to let it initialize with working sgl_kernel
print('[TT-Plugin] Starting SGLang with working sgl_kernel...')
import runpy
sys.argv = {repr(['python'] + cmd[1:])}

# Monkey-patch SGLang during startup
print('[TT-Plugin] Applying TT plugin monkey-patch...')
try:
    import sglang_tt_plugin
    success = sglang_tt_plugin.monkey_patch_sglang()
    if success:
        print('[TT-Plugin] Successfully patched SGLang with TT models')
    else:
        print('[TT-Plugin] Failed to patch SGLang, using default models')
except ImportError as e:
    print(f'[TT-Plugin] Failed to load TT plugin: {{e}}')

# Now run SGLang
runpy.run_module('sglang.launch_server', run_name='__main__')
"""
    ]
    
    # Launch server
    try:
        subprocess.run(python_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Server launch failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Server shutdown requested")
        sys.exit(0)

if __name__ == "__main__":
    main()
