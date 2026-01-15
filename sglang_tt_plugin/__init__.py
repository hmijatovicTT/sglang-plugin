"""
SGLang TT-Metal Plugin
"""

import logging
import os
import sys

# Set version at module level
__version__ = "0.1.0"

# CRITICAL: Set CPU-only environment variables FIRST
# These must be set before any PyTorch or SGLang imports
os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
os.environ["SGLANG_DISABLE_CUDA_KERNEL"] = "1"  # Updated from deprecated SGL_DISABLE_CUDA_KERNEL
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Prevent torch compilation issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

logger = logging.getLogger(__name__)

def register_tt_models():
    """Register TT-Metal models by overriding ModelRegistry.
    This must be called AFTER SGLang is imported."""
    import types
    
    try:
        import ttnn
        print("[TT-Plugin] TT-Metal (ttnn) is available", file=sys.stderr, flush=True)
        
        from .models.tt_llama import TTLlamaForCausalLM
        
        from sglang.srt.models.registry import ModelRegistry
        
        # Force registry population by trying to resolve a model
        # This ensures the registry is populated before we patch it
        try:
            ModelRegistry.resolve_model_cls(["LlamaForCausalLM"])
        except:
            pass  # Ignore errors, we just want to trigger population
        
        # Now overwrite the registry entry
        ModelRegistry.models["LlamaForCausalLM"] = TTLlamaForCausalLM
        
        # Also patch _try_load_model_cls to ensure it returns our class
        original_try_load = ModelRegistry._try_load_model_cls
        
        def patched_try_load(self, model_arch):
            if model_arch == "LlamaForCausalLM":
                print(f"[TT-Plugin] _try_load_model_cls intercepted for {model_arch}, returning TTLlamaForCausalLM", file=sys.stderr, flush=True)
                return TTLlamaForCausalLM
            return original_try_load(model_arch)
        
        ModelRegistry._try_load_model_cls = types.MethodType(patched_try_load, ModelRegistry)
        
        # Verify the patch
        test_result = ModelRegistry._try_load_model_cls("LlamaForCausalLM")
        if test_result is TTLlamaForCausalLM:
            print("[TT-Plugin] ✓ Successfully patched ModelRegistry - TTLlamaForCausalLM will be used", file=sys.stderr, flush=True)
        else:
            print(f"[TT-Plugin] ✗ Patch verification failed! Got: {test_result}", file=sys.stderr, flush=True)
        
        return True
        
    except ImportError as e:
        print(f"[TT-Plugin] TT-Metal not available: {e}", file=sys.stderr, flush=True)
        return False
    except Exception as e:
        print(f"[TT-Plugin] Error registering TT models: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return False

# Auto-register when SGLang is imported (for subprocess compatibility)
# This ensures the patch works in both main process and subprocesses
def _auto_register_if_sglang_imported():
    """Auto-register TT models if SGLang is already imported."""
    try:
        import sglang.srt.models.registry
        from sglang.srt.models.registry import ModelRegistry
        # Check if external package registration already happened
        if "LlamaForCausalLM" in ModelRegistry.models:
            current_cls = ModelRegistry.models["LlamaForCausalLM"]
            from .models.tt_llama import TTLlamaForCausalLM
            if current_cls is not TTLlamaForCausalLM:
                # External package didn't work, apply manual patch
                print("[TT-Plugin] External package registration didn't override, applying manual patch", file=sys.stderr, flush=True)
                register_tt_models()
            else:
                print("[TT-Plugin] External package registration successful", file=sys.stderr, flush=True)
        else:
            # Registry not populated yet, register manually
            register_tt_models()
    except ImportError:
        # SGLang not imported yet, that's fine
        pass

# Try to auto-register (will work if SGLang is already imported)
_auto_register_if_sglang_imported()

from .models.tt_llama import TTLlamaForCausalLM, TTModels
from .utils.tt_utils import open_mesh_device

__all__ = [
    "TTLlamaForCausalLM",
    "TTModels",
    "open_mesh_device", 
    "register_tt_models",
    "__version__",
]