#!/usr/bin/env python3
"""
Launch SGLang server with TT-Metal plugin support.
Based on the working command structure.
"""

import os
import sys
import subprocess
import argparse

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
    
    # Add TT plugin to Python path
    plugin_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if plugin_path not in sys.path:
        sys.path.insert(0, plugin_path)
    
    # Import TT plugin to register models
    try:
        import sglang_tt_plugin
        print("TT Plugin loaded successfully")
        print("TTLlamaForCausalLM should now be registered")
    except ImportError as e:
        print(f"Failed to load TT plugin: {e}")
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
    
    # Launch server
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Server launch failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Server shutdown requested")
        sys.exit(0)

if __name__ == "__main__":
    main()
