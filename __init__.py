"""SGLang TT Plugin for Tenstorrent devices."""
import logging

__version__ = "0.1.0"
_registered = False

def register_tt_models():
    """Register TT models with SGLang's ModelRegistry."""
    global _registered
    
    if _registered:
        return True
        
    logger = logging.getLogger(__name__)
    
    try:
        # Import SGLang's ModelRegistry - this should work if SGLang is already loaded
        from sglang.srt.models.registry import ModelRegistry
        
        # Check if we're already registered
        current_model = ModelRegistry.models.get("LlamaForCausalLM")
        if current_model and "TTLlama" in str(current_model):
            logger.info("[TT-Plugin] SGLang already has TT models registered")
            _registered = True
            return True
            
        # Import our TT model ONLY when actually needed for registration
        from .models.tt_llama import TTLlamaForCausalLM
        
        # Store the original for potential restoration
        original_model = ModelRegistry.models.get("LlamaForCausalLM")
        
        # Register the TT model
        ModelRegistry.models["LlamaForCausalLM"] = TTLlamaForCausalLM
        
        logger.info("[TT-Plugin] Successfully registered TT models with SGLang ModelRegistry")
        logger.info(f"[TT-Plugin] Replaced {original_model} with TTLlamaForCausalLM")
        
        _registered = True
        return True
        
    except ImportError as e:
        logger.warning(f"[TT-Plugin] Failed to register TT models: {e}")
        return False

# Export the registration function
__all__ = ["register_tt_models"]
