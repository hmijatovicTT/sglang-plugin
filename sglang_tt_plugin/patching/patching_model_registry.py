"""
This module handles the runtime patching of SGLang to use TT-Metal models.
"""
import sys
import logging
from sglang.srt.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

def register_tt_models():
    """Register TT-Metal models with SGLang's model registry."""
    print("[TT-Plugin] register_tt_models() called", file=sys.stderr, flush=True)
    try:
        # Import all TT model classes
        from ..models.tt_llm import (
            TTLlamaForCausalLM,
            TTQwenForCausalLM,
            TTMistralForCausalLM,
            TTGptOssForCausalLM,
        )
        print(f"[TT-Plugin] Imported TT model classes successfully", file=sys.stderr, flush=True)
        
        # Mapping from HuggingFace architecture names to TT model classes
        TT_MODEL_REGISTRY = {
            "LlamaForCausalLM": TTLlamaForCausalLM,      # Llama-3.1-8B, Llama-3.1-70B, etc.
            "Qwen2ForCausalLM": TTQwenForCausalLM,       # Qwen2.5-7B, Qwen2.5-14B, etc.
            "MistralForCausalLM": TTMistralForCausalLM,  # Mistral-7B
            "GptOssForCausalLM": TTGptOssForCausalLM,    # GPT-OSS
        }
        
        # CRITICAL: Directly patch SGLang's ModelRegistry
        ModelRegistry.models.update(TT_MODEL_REGISTRY)
        print(f"[TT-Plugin] ✓ Registered {len(TT_MODEL_REGISTRY)} TT models: {list(TT_MODEL_REGISTRY.keys())}", file=sys.stderr, flush=True)
        logger.info(f"[TT-Plugin] ✓ Registered {len(TT_MODEL_REGISTRY)} TT models")
            
    except Exception as e:
        print(f"[TT-Plugin] ERROR registering TT models: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        logger.error(f"[TT-Plugin] Error registering TT models: {e}")
