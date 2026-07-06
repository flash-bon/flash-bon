"""TaylorSeer-style caching + layer/timestep skipping for diffusion transformers."""

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from diffusers.utils import logging
from diffusers.hooks.hooks import HookRegistry, ModelHook, StateManager


logger = logging.get_logger(__name__)
_LAZY_BON_SCALING_HOOK = "lazy_bon_scaling"
_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = (
    "^blocks.*attn",
    "^transformer_blocks.*attn",
    "^single_transformer_blocks.*attn",
)
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = ("^temporal_transformer_blocks.*attn",)
_TRANSFORMER_BLOCK_IDENTIFIERS = (
    _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS + _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS
)
_BLOCK_IDENTIFIERS = ("^[^.]*block[^.]*\\.[^.]+$",)
_PROJ_OUT_IDENTIFIERS = ("^proj_out$",)

# Block-level patterns for layer skipping (matches whole blocks, not just attn).
# Flux has two block lists (transformer_blocks + single_transformer_blocks); Wan has one (blocks).
_DOUBLE_BLOCK_PATTERN = re.compile(r"^transformer_blocks\.(\d+)$")
_SINGLE_BLOCK_PATTERN = re.compile(r"^single_transformer_blocks\.(\d+)$")
_WAN_BLOCK_PATTERN = re.compile(r"^blocks\.(\d+)$")
# For matching in module iteration (without $ anchor)
_DOUBLE_BLOCK_MATCH = re.compile(r"^transformer_blocks\.(\d+)")
_SINGLE_BLOCK_MATCH = re.compile(r"^single_transformer_blocks\.(\d+)")
_WAN_BLOCK_MATCH = re.compile(r"^blocks\.(\d+)")


@dataclass
class LazyBonScalingConfig:
    """Configuration for the caching + skipping (based on TaylorSeer)."""

    cache_interval: int = 5
    disable_cache_before_step: int = 3
    disable_cache_after_step: Optional[int] = None
    max_order: int = 1
    taylor_factors_dtype: Optional[torch.dtype] = torch.bfloat16
    skip_predict_identifiers: Optional[List[str]] = None
    cache_identifiers: Optional[List[str]] = None
    use_lite_mode: bool = False
    # Layer/timestep skipping (all disabled by default for zero overhead)
    run_till_timestep: Optional[int] = None
    no_skip_till_timestep: Optional[int] = None
    skip_layer_start: Optional[int] = None
    skip_layer_end: Optional[int] = None
    resume_from_step: Optional[int] = None

    def __repr__(self) -> str:
        return (
            "LazyBonScalingConfig("
            f"cache_interval={self.cache_interval}, "
            f"disable_cache_before_step={self.disable_cache_before_step}, "
            f"disable_cache_after_step={self.disable_cache_after_step}, "
            f"max_order={self.max_order}, "
            f"taylor_factors_dtype={self.taylor_factors_dtype}, "
            f"skip_predict_identifiers={self.skip_predict_identifiers}, "
            f"cache_identifiers={self.cache_identifiers}, "
            f"use_lite_mode={self.use_lite_mode}, "
            f"run_till_timestep={self.run_till_timestep}, "
            f"no_skip_till_timestep={self.no_skip_till_timestep}, "
            f"skip_layer_start={self.skip_layer_start}, "
            f"skip_layer_end={self.skip_layer_end}, "
            f"resume_from_step={self.resume_from_step})"
        )


class LazyBonScalingState:
    def __init__(
        self,
        taylor_factors_dtype: Optional[torch.dtype] = torch.bfloat16,
        max_order: int = 1,
        is_inactive: bool = False,
    ):
        self.taylor_factors_dtype = taylor_factors_dtype
        self.max_order = max_order
        self.is_inactive = is_inactive

        self.module_dtypes: Tuple[torch.dtype, ...] = ()
        self.last_update_step: Optional[int] = None
        self.taylor_factors: Dict[int, Dict[int, torch.Tensor]] = {}
        self.inactive_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None
        self.device: Optional[torch.device] = None
        self.current_step: int = -1

    def reset(self) -> None:
        self.current_step = -1
        self.last_update_step = None
        self.taylor_factors = {}
        self.inactive_shapes = None
        self.device = None

    def update(
        self,
        outputs: Tuple[torch.Tensor, ...],
    ) -> None:
        self.module_dtypes = tuple(output.dtype for output in outputs)
        self.device = outputs[0].device

        if self.is_inactive:
            self.inactive_shapes = tuple(output.shape for output in outputs)
        else:
            for i, features in enumerate(outputs):
                new_factors: Dict[int, torch.Tensor] = {0: features}
                is_first_update = self.last_update_step is None
                if not is_first_update:
                    delta_step = self.current_step - self.last_update_step
                    if delta_step == 0:
                        raise ValueError(
                            "Delta step cannot be zero for TaylorSeer update."
                        )

                    # Recursive divided differences up to max_order
                    prev_factors = self.taylor_factors.get(i, {})
                    for j in range(self.max_order):
                        prev = prev_factors.get(j)
                        if prev is None:
                            break
                        new_factors[j + 1] = (
                            new_factors[j] - prev.to(features.dtype)
                        ) / delta_step
                self.taylor_factors[i] = {
                    order: factor.to(self.taylor_factors_dtype)
                    for order, factor in new_factors.items()
                }

        self.last_update_step = self.current_step

    @torch.compiler.disable
    def predict(self) -> List[torch.Tensor]:
        if self.last_update_step is None:
            raise ValueError("Cannot predict without prior initialization/update.")

        step_offset = self.current_step - self.last_update_step

        outputs = []
        if self.is_inactive:
            if self.inactive_shapes is None:
                raise ValueError("Inactive shapes not set during prediction.")
            for i in range(len(self.module_dtypes)):
                outputs.append(
                    torch.zeros(
                        self.inactive_shapes[i],
                        dtype=self.module_dtypes[i],
                        device=self.device,
                    )
                )
        else:
            if not self.taylor_factors:
                raise ValueError("Taylor factors empty during prediction.")
            num_outputs = len(self.taylor_factors)
            num_orders = len(self.taylor_factors[0])
            for i in range(num_outputs):
                output_dtype = self.module_dtypes[i]
                taylor_factors = self.taylor_factors[i]
                output = torch.zeros_like(taylor_factors[0], dtype=output_dtype)
                for order in range(num_orders):
                    coeff = (step_offset**order) / math.factorial(order)
                    factor = taylor_factors[order]
                    output = output + factor.to(output_dtype) * coeff
                outputs.append(output)
        return outputs

    @torch.compiler.disable
    def predict_frozen(self) -> List[torch.Tensor]:
        """Return 0th order factor (last computed output) without extrapolation."""
        outputs = []
        if self.is_inactive:
            for i in range(len(self.module_dtypes)):
                outputs.append(
                    torch.zeros(
                        self.inactive_shapes[i],
                        dtype=self.module_dtypes[i],
                        device=self.device,
                    )
                )
        else:
            for i in range(len(self.taylor_factors)):
                outputs.append(self.taylor_factors[i][0].to(self.module_dtypes[i]))
        return outputs


class LazyBonScalingHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        cache_interval: int,
        disable_cache_before_step: int,
        taylor_factors_dtype: torch.dtype,
        state_manager: StateManager,
        disable_cache_after_step: Optional[int] = None,
        run_till_timestep: Optional[int] = None,
        skip_after_step: Optional[
            int
        ] = None,  # Pre-computed: step after which to skip this layer
        is_single_block: bool = False,  # True = single_transformer_block (returns single tensor)
        returns_encoder_hidden_states: bool = False,  # True = block returns (encoder_hs, hs) tuple
    ):
        super().__init__()
        self.cache_interval = cache_interval
        self.disable_cache_before_step = disable_cache_before_step
        self.disable_cache_after_step = disable_cache_after_step
        self.taylor_factors_dtype = taylor_factors_dtype
        self.state_manager = state_manager
        self.run_till_timestep = run_till_timestep
        self.skip_after_step = skip_after_step
        self.is_single_block = is_single_block
        self.returns_encoder_hidden_states = returns_encoder_hidden_states

    def initialize_hook(self, module: torch.nn.Module):
        return module

    def reset_state(self, module: torch.nn.Module) -> None:
        """Reset state between sampling runs."""
        self.state_manager.reset()

    @torch.compiler.disable
    def _measure_should_compute(self) -> bool:
        state: LazyBonScalingState = self.state_manager.get_state()
        state.current_step += 1
        current_step = state.current_step
        is_warmup_phase = current_step < self.disable_cache_before_step
        is_compute_interval = (
            current_step - self.disable_cache_before_step - 1
        ) % self.cache_interval == 0
        is_cooldown_phase = (
            self.disable_cache_after_step is not None
            and current_step >= self.disable_cache_after_step
        )
        should_compute = is_warmup_phase or is_compute_interval or is_cooldown_phase
        return should_compute, state

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        should_compute, state = self._measure_should_compute()
        current_step = state.current_step

        if self.skip_after_step is not None and current_step >= self.skip_after_step:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and len(args) > 0:
                hidden_states = args[0]
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            if encoder_hidden_states is None and len(args) > 1:
                encoder_hidden_states = args[1]
            if self.returns_encoder_hidden_states:
                return (encoder_hidden_states, hidden_states)
            return hidden_states

        # Early stop: return frozen output at/after run_till_timestep
        if (
            self.run_till_timestep is not None
            and current_step >= self.run_till_timestep
        ):
            if state.last_update_step is not None:
                outputs_list = state.predict_frozen()
                return (
                    outputs_list[0] if len(outputs_list) == 1 else tuple(outputs_list)
                )
            # No cache yet - MUST compute to build cache, don't skip!
            should_compute = True

        if should_compute or state.last_update_step is None:
            outputs = self.fn_ref.original_forward(*args, **kwargs)
            wrapped_outputs = (
                (outputs,) if isinstance(outputs, torch.Tensor) else outputs
            )
            state.update(wrapped_outputs)
            return outputs

        outputs_list = state.predict()
        return outputs_list[0] if len(outputs_list) == 1 else tuple(outputs_list)


def _invalidate_child_registry_cache(module: torch.nn.Module) -> None:
    registry = getattr(module, "_diffusers_hook", None)
    if registry is not None:
        registry._child_registries_cache = None


def _resolve_patterns(config: LazyBonScalingConfig) -> Tuple[List[str], List[str]]:
    inactive_patterns = (
        config.skip_predict_identifiers
        if config.skip_predict_identifiers is not None
        else None
    )
    active_patterns = (
        config.cache_identifiers if config.cache_identifiers is not None else None
    )
    return inactive_patterns or [], active_patterns or []


def _extract_layer_index(
    module_name: str, num_double_blocks: int = 19
) -> Optional[int]:
    match = _DOUBLE_BLOCK_MATCH.match(module_name)
    if match:
        return int(match.group(1))
    match = _SINGLE_BLOCK_MATCH.match(module_name)
    if match:
        return int(match.group(1)) + num_double_blocks
    match = _WAN_BLOCK_MATCH.match(module_name)  # Wan: single `blocks.N` list
    if match:
        return int(match.group(1))
    return None


def apply_lazy_bon_scaling(module: torch.nn.Module, config: LazyBonScalingConfig):
    inactive_patterns, active_patterns = _resolve_patterns(config)

    active_patterns = active_patterns or _TRANSFORMER_BLOCK_IDENTIFIERS

    if config.use_lite_mode:
        logger.info("Using TaylorSeer Lite variant for cache.")
        active_patterns = _PROJ_OUT_IDENTIFIERS
        inactive_patterns = _BLOCK_IDENTIFIERS
        if config.skip_predict_identifiers or config.cache_identifiers:
            logger.warning("Lite mode overrides user patterns.")

    # Check if layer skip is enabled
    layer_skip_enabled = (
        config.no_skip_till_timestep is not None
        and config.skip_layer_start is not None
        and config.skip_layer_end is not None
    )

    # Count double blocks for layer indexing
    num_double_blocks = 19  # default
    if layer_skip_enabled:
        for name, _ in module.named_modules():
            match = _DOUBLE_BLOCK_PATTERN.match(name)
            if match:
                num_double_blocks = max(num_double_blocks, int(match.group(1)) + 1)

    # If layer skip enabled, add block-level patterns for the skip range (hooks the WHOLE block).
    skip_block_names = set()
    if layer_skip_enabled:
        for name, _ in module.named_modules():
            if (
                _DOUBLE_BLOCK_PATTERN.match(name)
                or _SINGLE_BLOCK_PATTERN.match(name)
                or _WAN_BLOCK_PATTERN.match(name)
            ):
                layer_idx = _extract_layer_index(name, num_double_blocks)
                if (
                    layer_idx is not None
                    and config.skip_layer_start <= layer_idx <= config.skip_layer_end
                ):
                    skip_block_names.add(name)

    for name, submodule in module.named_modules():
        is_skip_block = name in skip_block_names
        is_single_block = _SINGLE_BLOCK_PATTERN.match(name) is not None
        returns_encoder_hidden_states = (
            _DOUBLE_BLOCK_PATTERN.match(name) is not None
            or _SINGLE_BLOCK_PATTERN.match(name) is not None
        )

        matches_inactive = any(
            re.fullmatch(pattern, name) for pattern in inactive_patterns
        )
        matches_active = any(re.fullmatch(pattern, name) for pattern in active_patterns)

        if not (matches_inactive or matches_active or is_skip_block):
            continue

        skip_after_step = None
        if is_skip_block:
            skip_after_step = config.no_skip_till_timestep

        _apply_lazy_bon_scaling_hook(
            module=submodule,
            config=config,
            is_inactive=matches_inactive,
            skip_after_step=skip_after_step,
            is_single_block=is_single_block,
            returns_encoder_hidden_states=returns_encoder_hidden_states,
        )

    _invalidate_child_registry_cache(module)


def _apply_lazy_bon_scaling_hook(
    module: nn.Module,
    config: LazyBonScalingConfig,
    is_inactive: bool,
    skip_after_step: Optional[int] = None,
    is_single_block: bool = False,
    returns_encoder_hidden_states: bool = False,
):
    """Register the caching hook on the specified nn.Module."""
    state_manager = StateManager(
        LazyBonScalingState,
        init_kwargs={
            "taylor_factors_dtype": config.taylor_factors_dtype,
            "max_order": config.max_order,
            "is_inactive": is_inactive,
        },
    )

    registry = HookRegistry.check_if_exists_or_initialize(module)

    hook = LazyBonScalingHook(
        cache_interval=config.cache_interval,
        disable_cache_before_step=config.disable_cache_before_step,
        taylor_factors_dtype=config.taylor_factors_dtype,
        disable_cache_after_step=config.disable_cache_after_step,
        run_till_timestep=config.run_till_timestep,
        skip_after_step=skip_after_step,
        is_single_block=is_single_block,
        returns_encoder_hidden_states=returns_encoder_hidden_states,
        state_manager=state_manager,
    )

    registry.register_hook(hook, _LAZY_BON_SCALING_HOOK)


def reset_lazy_bon_scaling(module: torch.nn.Module):
    for submodule in module.modules():
        registry = getattr(submodule, "_diffusers_hook", None)
        if registry is None:
            continue
        hook = registry.get_hook(_LAZY_BON_SCALING_HOOK)
        if hook is not None:
            hook.reset_state(submodule)


def remove_lazy_bon_scaling(module: torch.nn.Module):
    for submodule in module.modules():
        registry = getattr(submodule, "_diffusers_hook", None)
        if registry is None:
            continue
        if registry.get_hook(_LAZY_BON_SCALING_HOOK) is not None:
            registry.remove_hook(_LAZY_BON_SCALING_HOOK, recurse=False)
    _invalidate_child_registry_cache(module)
