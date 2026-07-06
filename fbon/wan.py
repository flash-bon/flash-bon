"""FlashBonWanPipeline: stock ``diffusers.WanPipeline`` + stateful draft/resume (T2I mode).

A thin SUBCLASS of the stock Wan2.1 pipeline:

  * ``seed_list``       -- per-sample seeds for a batch of drafts
  * ``capture_at_step`` -- snapshot generation state into a ``GenerationCache`` at this step
  * ``generation_cache`` / ``resume_from_step`` -- resume from a (sliced) cache to full quality

Wan is a video model; we use it as a text-to-image generator with ``num_frames=1`` and return
the single decoded frame as a PIL image, so the verifier / eval-metric / budget-loop code is
identical across Flux and Wan. The denoising body mirrors diffusers' ``WanPipeline.__call__``
(CFG via two transformer passes, 5-D latents, flow-matching scheduler); the only additions are
the capture hook and the resume dispatch.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import WanPipeline

from .cache_state import GenerationCache


@dataclass
class FlashBonWanOutput:
    """Same shape as FlashBonFluxOutput so downstream code is model-agnostic."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    generation_cache: Optional[GenerationCache] = None


def _seeds_to_generators(seed_list, device):
    if seed_list is None:
        return None
    return [torch.Generator(device=device).manual_seed(int(s)) for s in seed_list]


class FlashBonWanPipeline(WanPipeline):
    """Wan2.1 pipeline with capture/resume for the draft -> select -> refine workflow (T2I)."""

    def _frames_to_pil(self, video) -> List[Image.Image]:
        """Decoded Wan output -> one PIL image per batch item (the single T2I frame)."""
        pil = self.video_processor.postprocess_video(video, output_type="pil")
        # postprocess_video returns a list (batch) of lists (frames); take frame 0 of each.
        return [
            frames[0] if isinstance(frames, (list, tuple)) else frames for frames in pil
        ]

    def _decode(self, latents) -> List[Image.Image]:
        latents = latents.to(self.vae.dtype)
        mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / inv_std + mean
        video = self.vae.decode(latents, return_dict=False)[0]
        return self._frames_to_pil(video)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: int = 1,
        generator=None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        return_dict: bool = True,
        # ---- stateful resumption ----
        capture_at_step: Optional[int] = None,
        generation_cache: Optional[GenerationCache] = None,
        seed_list: Optional[List[int]] = None,
        resume_from_step: Optional[int] = None,
    ):
        if generation_cache is not None:
            return self._resume_from_cache(
                generation_cache, resume_from_step, return_dict
            )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if seed_list is not None:
            if generator is not None:
                raise ValueError("Pass either `generator` or `seed_list`, not both.")
            generator = _seeds_to_generators(seed_list, device)

        # 3. Encode prompt (CFG => also encode negative)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Latents (5-D: B, C, T, H, W)
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        do_cfg = self.do_classifier_free_guidance
        self._num_timesteps = len(timesteps)
        self.scheduler.set_begin_index(0)
        captured_cache = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                if do_cfg:
                    with self.transformer.cache_context("uncond"):
                        noise_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + guidance_scale * (
                        noise_pred - noise_uncond
                    )

                # ---- capture state for resumption ----
                if capture_at_step is not None and i == capture_at_step:
                    captured_cache = GenerationCache(
                        latents=latents.clone(),
                        step_index=i,
                        prompt_embeds=prompt_embeds.clone(),
                        pooled_prompt_embeds=None,
                        text_ids=None,
                        latent_image_ids=None,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        batch_size=latents.shape[0],
                        timesteps=timesteps.clone(),
                        negative_prompt_embeds=negative_prompt_embeds.clone()
                        if do_cfg
                        else None,
                        seed_list=list(seed_list) if seed_list is not None else None,
                        noise_pred=noise_pred.clone(),
                        timestep_value=t.clone(),
                        extra={"num_frames": num_frames},
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
                progress_bar.update()

        self._current_timestep = None
        images = self._decode(latents)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (images, captured_cache)
        return FlashBonWanOutput(images=images, generation_cache=captured_cache)

    def _resume_from_cache(
        self, cache: GenerationCache, resume_from_step: Optional[int], return_dict: bool
    ):
        device = self._execution_device
        latents = cache.latents.to(device)
        cached_noise_pred = cache.noise_pred.to(device)
        if resume_from_step is None:
            resume_from_step = cache.step_index + 1

        # Reset the scheduler so its multistep history (e.g. UniPC model_outputs from the
        # batch-sized draft run) is cleared -- otherwise resuming a *sliced* (smaller) batch
        # mismatches the retained state. set_timesteps is deterministic, so the schedule matches.
        self.scheduler.set_timesteps(cache.num_inference_steps, device=device)
        timesteps = cache.timesteps.to(device)
        prompt_embeds = cache.prompt_embeds.to(device)
        negative_prompt_embeds = (
            cache.negative_prompt_embeds.to(device)
            if cache.negative_prompt_embeds is not None
            else None
        )
        guidance_scale = cache.guidance_scale
        do_cfg = negative_prompt_embeds is not None and guidance_scale > 1.0
        transformer_dtype = self.transformer.dtype

        self._guidance_scale = guidance_scale
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False

        cached_step_index = cache.step_index
        t_cached = timesteps[cached_step_index]
        if resume_from_step <= cached_step_index:
            # backward extrapolation (add noise back via inverse flow step)
            t_target = timesteps[resume_from_step]
            dt = t_target - t_cached
            latents = latents + dt * cached_noise_pred
        elif resume_from_step == cached_step_index + 1:
            # forward bridge: apply cached prediction to advance exactly one step
            self.scheduler._step_index = cached_step_index
            latents = self.scheduler.step(
                cached_noise_pred, t_cached, latents, return_dict=False
            )[0]

        self.scheduler._step_index = resume_from_step
        remaining = timesteps[resume_from_step:]
        with self.progress_bar(total=len(remaining)) as progress_bar:
            for t in remaining:
                if self.interrupt:
                    continue
                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                if do_cfg:
                    with self.transformer.cache_context("uncond"):
                        noise_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + guidance_scale * (
                        noise_pred - noise_uncond
                    )
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
                progress_bar.update()

        self._current_timestep = None
        images = self._decode(latents)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (images, None)
        return FlashBonWanOutput(images=images, generation_cache=None)
