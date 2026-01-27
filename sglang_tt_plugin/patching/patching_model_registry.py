"""
This module handles the runtime patching of SGLang to use TT-Metal models.
"""
import logging
from sglang.srt.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

def register_tt_models():
    """Register TT-Metal models with SGLang's model registry."""
    try:
        # Check if TT-Metal is available
        import ttnn
        logger.info("[TT-Plugin] TT-Metal (ttnn) is available")
        
        # Import all TT model classes
        from ..models.tt_llm import (
            TTLlamaForCausalLM,
            TTQwenForCausalLM,
            TTMistralForCausalLM,
            TTGptOssForCausalLM,
        )
        
        # Mapping from HuggingFace architecture names to TT model classes
        TT_MODEL_REGISTRY = {
            "LlamaForCausalLM": TTLlamaForCausalLM,      # Llama-3.1-8B, Llama-3.1-70B, etc.
            "Qwen2ForCausalLM": TTQwenForCausalLM,       # Qwen2.5-7B, Qwen2.5-14B, etc.
            "MistralForCausalLM": TTMistralForCausalLM,  # Mistral-7B
            "GptOssForCausalLM": TTGptOssForCausalLM,    # GPT-OSS
        }
        
        # CRITICAL: Directly patch SGLang's ModelRegistry
        try:
            # Override all supported architectures in the registry
            for arch_name, tt_class in TT_MODEL_REGISTRY.items():
                ModelRegistry.models[arch_name] = tt_class
                logger.info(f"[TT-Plugin] Registered {arch_name} -> {tt_class.__name__}")
            
            # Also patch _try_load_model_cls to intercept model loading
            original_try_load = ModelRegistry._try_load_model_cls
            
            @staticmethod
            def patched_try_load_model_cls(architectures):
                for arch in architectures:
                    if arch in TT_MODEL_REGISTRY:
                        tt_class = TT_MODEL_REGISTRY[arch]
                        logger.info(f"[TT-Plugin] Intercepted load for {arch}, returning {tt_class.__name__}")
                        return tt_class
                return original_try_load(architectures)
            
            ModelRegistry._try_load_model_cls = patched_try_load_model_cls
            logger.info(f"[TT-Plugin] ✓ Successfully patched ModelRegistry with {len(TT_MODEL_REGISTRY)} TT models in patch")
            
        except ImportError as e:
            logger.warning(f"[TT-Plugin] Could not import ModelRegistry: {e}")
        
    except ImportError as e:
        logger.warning(f"[TT-Plugin] TT-Metal not available: {e}")
    except Exception as e:
        logger.error(f"[TT-Plugin] Error registering TT models: {e}")
