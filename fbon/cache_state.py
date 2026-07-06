"""Stateful resumption support for the draft -> select -> refine workflow.

A :class:`GenerationCache` captures everything needed to resume denoising from a specific
step. The workflow is:

    1. Generate a batch of cheap drafts with early stopping, capturing state.
    2. Score the drafts, select the best.
    3. Resume generation (full quality) for the selected draft(s) only.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GenerationCache:
    """All state needed to resume diffusion denoising from a captured step."""

    latents: torch.Tensor
    step_index: int
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor
    latent_image_ids: torch.Tensor
    guidance_scale: float
    height: int
    width: int
    num_inference_steps: int
    batch_size: int
    timesteps: Optional[torch.Tensor] = None
    # Optional: for true CFG support
    negative_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None
    negative_text_ids: Optional[torch.Tensor] = None
    # Optional: seed_list for reproducibility
    seed_list: Optional[List[int]] = None
    # Optional: noise_pred for bridging/back-extrapolating the captured step during resume
    noise_pred: Optional[torch.Tensor] = None
    # Timestep value at capture (for the manual scheduler bridge step)
    timestep_value: Optional[torch.Tensor] = None
    # Model-specific extras (non-batched scalars), e.g. {"num_frames": 1} for Wan.
    extra: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "GenerationCache":
        return GenerationCache(
            latents=self.latents.clone(),
            step_index=self.step_index,
            prompt_embeds=self.prompt_embeds.clone(),
            pooled_prompt_embeds=self.pooled_prompt_embeds.clone(),
            text_ids=self.text_ids.clone(),
            latent_image_ids=self.latent_image_ids.clone()
            if self.latent_image_ids is not None
            else None,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            batch_size=self.batch_size,
            timesteps=self.timesteps.clone() if self.timesteps is not None else None,
            negative_prompt_embeds=self.negative_prompt_embeds.clone()
            if self.negative_prompt_embeds is not None
            else None,
            negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds.clone()
            if self.negative_pooled_prompt_embeds is not None
            else None,
            negative_text_ids=self.negative_text_ids.clone()
            if self.negative_text_ids is not None
            else None,
            seed_list=list(self.seed_list) if self.seed_list is not None else None,
            noise_pred=self.noise_pred.clone() if self.noise_pred is not None else None,
            timestep_value=self.timestep_value.clone()
            if self.timestep_value is not None
            else None,
            extra=dict(self.extra),
        )

    def to(self, device: torch.device) -> "GenerationCache":
        return GenerationCache(
            latents=self.latents.to(device),
            step_index=self.step_index,
            prompt_embeds=self.prompt_embeds.to(device),
            pooled_prompt_embeds=self.pooled_prompt_embeds.to(device),
            text_ids=self.text_ids.to(device),
            latent_image_ids=self.latent_image_ids.to(device)
            if self.latent_image_ids is not None
            else None,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            batch_size=self.batch_size,
            timesteps=self.timesteps.to(device) if self.timesteps is not None else None,
            negative_prompt_embeds=self.negative_prompt_embeds.to(device)
            if self.negative_prompt_embeds is not None
            else None,
            negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds.to(device)
            if self.negative_pooled_prompt_embeds is not None
            else None,
            negative_text_ids=self.negative_text_ids.to(device)
            if self.negative_text_ids is not None
            else None,
            seed_list=self.seed_list,
            noise_pred=self.noise_pred.to(device)
            if self.noise_pred is not None
            else None,
            timestep_value=self.timestep_value.to(device)
            if self.timestep_value is not None
            else None,
            extra=dict(self.extra),
        )


def slice_cache(
    cache: Optional[GenerationCache], indices: List[int]
) -> Optional[GenerationCache]:
    """Keep only the given batch indices of a cache (e.g. select the best draft)."""
    if cache is None:
        return None

    batch_size = cache.batch_size
    device = cache.latents.device
    indices_tensor = torch.tensor(indices, device=device)

    def maybe_slice(tensor: Optional[torch.Tensor], bs: int) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.dim() >= 1 and tensor.size(0) == bs:
            return tensor[indices_tensor]
        return tensor

    sliced_seed_list = None
    if cache.seed_list is not None:
        sliced_seed_list = [cache.seed_list[i] for i in indices]

    return GenerationCache(
        latents=maybe_slice(cache.latents, batch_size),
        step_index=cache.step_index,
        prompt_embeds=maybe_slice(cache.prompt_embeds, batch_size),
        pooled_prompt_embeds=maybe_slice(cache.pooled_prompt_embeds, batch_size),
        text_ids=maybe_slice(cache.text_ids, batch_size),
        latent_image_ids=cache.latent_image_ids,  # not batched
        guidance_scale=cache.guidance_scale,
        height=cache.height,
        width=cache.width,
        num_inference_steps=cache.num_inference_steps,
        batch_size=len(indices),
        timesteps=cache.timesteps,  # not batched
        negative_prompt_embeds=maybe_slice(cache.negative_prompt_embeds, batch_size),
        negative_pooled_prompt_embeds=maybe_slice(
            cache.negative_pooled_prompt_embeds, batch_size
        ),
        negative_text_ids=maybe_slice(cache.negative_text_ids, batch_size),
        seed_list=sliced_seed_list,
        noise_pred=maybe_slice(cache.noise_pred, batch_size),
        timestep_value=cache.timestep_value,  # not batched
        extra=dict(cache.extra),
    )


def merge_caches(caches: List[GenerationCache]) -> GenerationCache:
    """Concatenate several caches (same step_index/dims) into one batched cache."""
    if not caches:
        raise ValueError("Cannot merge empty list of caches")
    if len(caches) == 1:
        return caches[0].clone()

    reference = caches[0]
    for c in caches[1:]:
        if c.step_index != reference.step_index:
            raise ValueError("All caches must have the same step_index")
        if c.height != reference.height or c.width != reference.width:
            raise ValueError("All caches must have the same dimensions")
        if c.num_inference_steps != reference.num_inference_steps:
            raise ValueError("All caches must have the same num_inference_steps")

    def cat_field(getter) -> Optional[torch.Tensor]:
        """Concatenate a field along the batch dim ONLY if it is actually batched (dim0 == batch_size)
        in every cache; otherwise it is a SHARED/unbatched tensor (e.g. FLUX ``text_ids`` /
        ``latent_image_ids`` position ids, which are per-sequence not per-sample) and we keep the
        reference's copy -- concatenating those would over-length the RoPE / attention sequence."""
        vals = [getter(c) for c in caches]
        if any(v is None for v in vals):
            return getter(reference)
        if all(
            v.dim() >= 1 and v.size(0) == c.batch_size for v, c in zip(vals, caches)
        ):
            return torch.cat(vals, dim=0)
        return getter(reference)  # shared/unbatched

    total_batch_size = sum(c.batch_size for c in caches)

    merged_seed_list = None
    if all(c.seed_list is not None for c in caches):
        merged_seed_list = []
        for c in caches:
            merged_seed_list.extend(c.seed_list)

    return GenerationCache(
        latents=cat_field(lambda c: c.latents),
        step_index=reference.step_index,
        prompt_embeds=cat_field(lambda c: c.prompt_embeds),
        pooled_prompt_embeds=cat_field(lambda c: c.pooled_prompt_embeds),
        text_ids=cat_field(lambda c: c.text_ids),
        latent_image_ids=reference.latent_image_ids,
        guidance_scale=reference.guidance_scale,
        height=reference.height,
        width=reference.width,
        num_inference_steps=reference.num_inference_steps,
        batch_size=total_batch_size,
        timesteps=reference.timesteps,
        negative_prompt_embeds=cat_field(lambda c: c.negative_prompt_embeds),
        negative_pooled_prompt_embeds=cat_field(
            lambda c: c.negative_pooled_prompt_embeds
        ),
        negative_text_ids=cat_field(lambda c: c.negative_text_ids),
        seed_list=merged_seed_list,
        noise_pred=cat_field(lambda c: c.noise_pred),
        timestep_value=reference.timestep_value,
        extra=dict(reference.extra),
    )
