"""Budget-constrained verifier-guided generation for diffusion models (Flash-BoN).

Draft cheap candidates with caching/layer-skipping, select the best with a pairwise VLM
tournament, then refine the top-k to full quality by resuming denoising from a captured state.
"""

from .cache_state import GenerationCache, merge_caches, slice_cache
from .caching import (
    apply_lazy_bon_scaling,
    remove_lazy_bon_scaling,
    reset_lazy_bon_scaling,
)
from .config import load_lazybon_config
from .models import get_model_spec, load_pipeline
from .verifier import PairwiseRankingVerifier
from .vqascore import VqascoreMetric

__all__ = [
    "GenerationCache", "merge_caches", "slice_cache",
    "apply_lazy_bon_scaling", "remove_lazy_bon_scaling", "reset_lazy_bon_scaling",
    "load_lazybon_config", "get_model_spec", "load_pipeline",
    "PairwiseRankingVerifier", "VqascoreMetric",
]
