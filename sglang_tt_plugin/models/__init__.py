"""TT model implementations for SGLang."""

import sys
import os

# Add sglang source code to Python path FIRST
SGLANG_SOURCE_PATH = "/localdev/hmijatovic/sglang/python"
if SGLANG_SOURCE_PATH not in sys.path:
    sys.path.insert(0, SGLANG_SOURCE_PATH)

from .tt_llama import TTLlamaForCausalLM, TTModels

__all__ = ["TTLlamaForCausalLM", "TTModels"]