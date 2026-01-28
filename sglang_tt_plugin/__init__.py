"""SGLang TT-Metal Plugin."""
import sys
print("[TT-Plugin] __init__.py loading...", file=sys.stderr, flush=True)

from .patching.patching_model_registry import register_tt_models

__version__ = "0.1.0"

register_tt_models()  # patch SGLang ModelRegistry on import
print("[TT-Plugin] __init__.py complete", file=sys.stderr, flush=True)
