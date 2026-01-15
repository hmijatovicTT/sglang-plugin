"""
SGLang TT-Metal Plugin

This plugin monkey-patches SGLang's LlamaForCausalLM to use TT-Metal's TTLlamaForCausalLM
which calls TTNN for hardware acceleration.
"""

import logging
import os
import sys

# Add sglang source code to Python path FIRST
SGLANG_SOURCE_PATH = "/localdev/hmijatovic/sglang/python"
if SGLANG_SOURCE_PATH not in sys.path:
    sys.path.insert(0, SGLANG_SOURCE_PATH)

# Set version at module level
__version__ = "0.1.0"

# CRITICAL: Set CPU engine environment variable BEFORE any SGLang imports
# This ensures sgl_kernel uses CPU operations, not GPU
os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
os.environ["SGL_DISABLE_CUDA_KERNEL"] = "1" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Prevent torch compilation issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

logger = logging.getLogger(__name__)

def register_tt_models():
    """Register TT-Metal models with SGLang's model registry."""
    try:
        # Check if TT-Metal is available
        import ttnn
        logger.info("[TT-Plugin] TT-Metal (ttnn) is available")
        
        # CRITICAL: Import and patch BEFORE SGLang can load its own modules
        # This must happen before any SGLang imports
        from .models.tt_llama import TTLlamaForCausalLM
        
        # Method 1: Register with SGLang's ModelRegistry (preferred method)
        try:
            # Try to import and register with ModelRegistry if SGLang is available
            from sglang.srt.models.registry import ModelRegistry
            # Register our model package - use overwrite=True to ensure our model takes precedence
            ModelRegistry.register("sglang_tt_plugin.models", overwrite=True, strict=False)
            logger.info("[TT-Plugin] Registered sglang_tt_plugin.models package with SGLang ModelRegistry")
            
            # Also directly override LlamaForCausalLM in the registry to ensure it's used
            ModelRegistry.models["LlamaForCausalLM"] = TTLlamaForCausalLM
            logger.info("[TT-Plugin] Overwriting LlamaForCausalLM with TTLlamaForCausalLM in ModelRegistry")
        except ImportError:
            # SGLang not yet imported, will register later via _ensure_registered_with_model_registry
            logger.debug("[TT-Plugin] SGLang ModelRegistry not yet available, will register on first SGLang import")
        except Exception as e:
            logger.warning(f"[TT-Plugin] Could not register with ModelRegistry: {e}")
        
        # Method 2: Preemptively patch sys.modules (fallback/backup method)
        import types
        fake_tt_module = types.ModuleType('tt_llama')
        fake_tt_module.TTLlamaForCausalLM = TTLlamaForCausalLM
        fake_tt_module.LlamaForCausalLM = TTLlamaForCausalLM  # Override both
        fake_tt_module.EntryClass = [TTLlamaForCausalLM]  # Match SGLang's pattern
        
        # Patch all possible module paths BEFORE SGLang loads them
        module_paths = [
            'sglang.srt.models.tt_llama',
            'tt_llama',
        ]
        
        for module_path in module_paths:
            sys.modules[module_path] = fake_tt_module
            logger.info(f"[TT-Plugin] Pre-patched {module_path}")
        
        # Method 3: Hook the model loader directly (additional safety)
        try:
            import importlib.util
            original_find_spec = importlib.util.find_spec
            
            def patched_find_spec(name, package=None):
                if name == 'sglang.srt.models.tt_llama' or name.endswith('.tt_llama'):
                    logger.debug(f"[TT-Plugin] INTERCEPTED find_spec for {name}")
                    # Return our fake module spec
                    spec = importlib.util.spec_from_loader(name, loader=None)
                    return spec
                return original_find_spec(name, package)
            
            importlib.util.find_spec = patched_find_spec
            
        except Exception as e:
            logger.warning(f"[TT-Plugin] Could not patch importlib: {e}")
        
        logger.info("[TT-Plugin] Successfully registered TT-Metal models")
        
    except ImportError as e:
        logger.warning(f"[TT-Plugin] TT-Metal not available: {e}")
    except Exception as e:
        logger.error(f"[TT-Plugin] Error registering TT models: {e}")

# CRITICAL: Register IMMEDIATELY on import, before SGLang can load anything
register_tt_models()

# Lazy registration function that can be called when SGLang is imported
def _ensure_registered_with_model_registry():
    """Ensure we're registered with ModelRegistry when SGLang loads."""
    try:
        from sglang.srt.models.registry import ModelRegistry
        from .models.tt_llama import TTLlamaForCausalLM
        
        # Register package if not already registered - use overwrite=True to ensure precedence
        if "LlamaForCausalLM" not in ModelRegistry.models or \
           ModelRegistry.models["LlamaForCausalLM"] != TTLlamaForCausalLM:
            ModelRegistry.register("sglang_tt_plugin.models", overwrite=True, strict=False)
            ModelRegistry.models["LlamaForCausalLM"] = TTLlamaForCausalLM
            logger.info("[TT-Plugin] Registered with ModelRegistry (lazy registration)")
            return True
    except Exception as e:
        logger.debug(f"[TT-Plugin] ModelRegistry not yet available: {e}")
    return False

# Try to register immediately if SGLang is already imported
_ensure_registered_with_model_registry()

# Hook into SGLang's registry module import using importlib
# This is safer than patching __builtins__
try:
    import importlib
    import importlib.util
    
    # Store original import_module
    _original_import_module = importlib.import_module
    
    def _patched_import_module(name, package=None):
        """Intercept module imports to register when SGLang's registry loads."""
        module = _original_import_module(name, package)
        
        # When SGLang's registry is imported, ensure we're registered
        if name == "sglang.srt.models.registry":
            _ensure_registered_with_model_registry()
        
        return module
    
    # Patch importlib.import_module
    importlib.import_module = _patched_import_module
except Exception as e:
    logger.debug(f"[TT-Plugin] Could not patch importlib: {e}")

from .models.tt_llama import TTLlamaForCausalLM, TTModels
from .utils.tt_utils import open_mesh_device

__all__ = [
    "TTLlamaForCausalLM",
    "TTModels",
    "open_mesh_device", 
    "register_tt_models",
    "__version__",
]