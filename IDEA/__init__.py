"""Public API for the refactored IDEA modules."""

from .model import init_model
from .niche import init_model as init_niche_model

# Backward compatible: init_model() remains IDEA-C.
init_deconv_model = init_model

__all__ = ["init_model", "init_deconv_model", "init_niche_model"]
