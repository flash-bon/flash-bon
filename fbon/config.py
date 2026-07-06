import json
from typing import Dict, Optional, Tuple

import torch

from .caching import LazyBonScalingConfig


def build_lazybon_config(
    cache_freq: int = 5,
    no_skip_till_timestep: int = 10,
    skip_start_layer: Optional[int] = None,
    skip_end_layer: Optional[int] = None,
    only_first_n_timesteps: Optional[int] = None,
    sample_steps: int = 50,
    use_lite_mode: bool = True,
) -> LazyBonScalingConfig:
    """Build a ``LazyBonScalingConfig`` from individual knobs (the ``optimal_params`` fields).

    Layer skip is only enabled when both ``skip_start_layer`` and ``skip_end_layer`` are provided.
    """
    run_till = (
        int(only_first_n_timesteps)
        if only_first_n_timesteps is not None
        else sample_steps
    )
    # Negative bounds (e.g. -1) are the "layer skip disabled" sentinel, same as null.
    skip_start = (
        int(skip_start_layer)
        if skip_start_layer is not None and int(skip_start_layer) >= 0
        else None
    )
    skip_end = (
        int(skip_end_layer)
        if skip_end_layer is not None and int(skip_end_layer) >= 0
        else None
    )

    return LazyBonScalingConfig(
        cache_interval=int(cache_freq),
        max_order=1,
        disable_cache_before_step=int(no_skip_till_timestep),
        disable_cache_after_step=sample_steps - 2,
        taylor_factors_dtype=torch.bfloat16,
        use_lite_mode=bool(use_lite_mode),
        no_skip_till_timestep=int(no_skip_till_timestep),
        skip_layer_start=skip_start,
        skip_layer_end=skip_end,
        run_till_timestep=run_till,
        resume_from_step=run_till,
    )


def load_lazybon_config(
    config_path: str,
    sample_steps: int = 50,
) -> Tuple[Optional[LazyBonScalingConfig], Dict]:
    """Build a ``LazyBonScalingConfig`` from a JSON file.

    Returns ``(config, base_config)``. ``config`` is ``None`` for baseline configs with no
    ``optimal_params`` (stock generation). ``config.resume_from_step`` is set to the step the
    draft->refine resumes from.
    """
    with open(config_path, "r") as f:
        base_config = json.load(f)

    optimal_params = base_config.get("optimal_params", None)
    if optimal_params is None:
        return None, base_config

    run_till_timestep = int(optimal_params.get("only_first_n_timesteps", sample_steps))

    config = build_lazybon_config(
        cache_freq=optimal_params.get("cache_freq", 5),
        no_skip_till_timestep=optimal_params.get("no_skip_till_timestep", 10),
        skip_start_layer=optimal_params.get("skip_start_layer", None),
        skip_end_layer=optimal_params.get("skip_end_layer", None),
        only_first_n_timesteps=optimal_params.get("only_first_n_timesteps", None),
        sample_steps=sample_steps,
        use_lite_mode=bool(base_config.get("use_lite_mode", True)),
    )

    # The refine RESUMES from this step (re-denoises [resume, end] at full quality). Larger offset
    # => earlier resume => deeper, higher-quality refinement.
    offset = int(base_config["cache_for_full_offset"])
    config.resume_from_step = min(run_till_timestep, sample_steps - offset)

    return config, base_config
