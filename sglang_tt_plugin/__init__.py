"""SGLang TT-Metal Plugin."""

from .patching.patching_model_registry import register_tt_models

__version__ = "0.1.0"

register_tt_models()

from .models.tt_llm import (
    TTModels,
    TTLlamaForCausalLM,
    TTQwenForCausalLM,
    TTMistralForCausalLM,
    TTGptOssForCausalLM,
    EntryClass,
)
from .utils.tt_utils import open_mesh_device, close_mesh_device, get_mesh_grid

__all__ = [
    "TTModels",
    "TTLlamaForCausalLM",
    "TTQwenForCausalLM",
    "TTMistralForCausalLM",
    "TTGptOssForCausalLM",
    "EntryClass",
    "open_mesh_device",
    "close_mesh_device",
    "get_mesh_grid",
    "register_tt_models",
    "__version__",
]