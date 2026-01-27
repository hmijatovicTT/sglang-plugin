"""SGLang TT-Metal Plugin."""

from contextlib import suppress
import importlib.util
import logging
import sys
from pathlib import Path


def _ensure_tt_metal_on_path() -> None:
    """Ensure the TT-Metal repo (providing `models.*`) is importable."""

    with suppress(ModuleNotFoundError):
        import models.tt_transformers.tt.generator_sglang  # noqa: F401
        return

    spec = importlib.util.find_spec("ttnn")
    if spec and spec.origin:
        tt_metal_root = Path(spec.origin).resolve().parents[2]
        if tt_metal_root.exists() and str(tt_metal_root) not in sys.path:
            sys.path.insert(0, str(tt_metal_root))

    try:
        import models.tt_transformers.tt.generator_sglang  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        logging.getLogger(__name__).warning(
            "[TT-Plugin] Could not locate TT-Metal repo on PYTHONPATH."
        )
        raise exc


_ensure_tt_metal_on_path()

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