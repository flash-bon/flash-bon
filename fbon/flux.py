"""FlashBonFluxPipeline: stock ``diffusers.FluxPipeline`` + stateful draft/resume.

A thin SUBCLASS of the stock pipeline. It adds three things to ``__call__``:

  * ``seed_list``        -- per-sample seeds for a batch of drafts (built into a generator list)
  * ``capture_at_step``  -- snapshot generation state into a ``GenerationCache`` at this step
  * ``generation_cache`` / ``resume_from_step`` -- resume from a (sliced) cache
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.utils import logging

from .cache_state import GenerationCache


logger = logging.get_logger(__name__)


@dataclass
class FlashBonFluxOutput:
    """Pipeline output carrying images plus an optional captured generation cache."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    generation_cache: Optional[GenerationCache] = None


def _seeds_to_generators(
    seed_list: Optional[List[int]], device
) -> Optional[List[torch.Generator]]:
    if seed_list is None:
        return None
    return [torch.Generator(device=device).manual_seed(int(s)) for s in seed_list]


class FlashBonFluxPipeline(FluxPipeline):
    """Flux pipeline with capture/resume for the draft -> select -> refine workflow."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # ---- stateful resumption ----
        capture_at_step: Optional[int] = None,
        generation_cache: Optional[GenerationCache] = None,
        seed_list: Optional[List[int]] = None,
        resume_from_step: Optional[int] = None,
    ):
        # Dispatch to resume path if a cache is supplied.
        if generation_cache is not None:
            return self._resume_from_cache(
                generation_cache=generation_cache,
                resume_from_step=resume_from_step,
                output_type=output_type,
                return_dict=return_dict,
                joint_attention_kwargs=joint_attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # seed_list -> per-sample generators (stock prepare_latents accepts a generator list)
        if seed_list is not None:
            if generator is not None:
                raise ValueError("Pass either `generator` or `seed_list`, not both.")
            generator = _seeds_to_generators(seed_list, device)

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None
            and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        negative_text_ids = None
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        if (
            hasattr(self.scheduler.config, "use_flow_sigmas")
            and self.scheduler.config.use_flow_sigmas
        ):
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        captured_cache = None

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
                    )

                # ---- capture state for resumption ----
                if capture_at_step is not None and i == capture_at_step:
                    captured_cache = GenerationCache(
                        latents=latents.clone(),
                        step_index=i,
                        prompt_embeds=prompt_embeds.clone(),
                        pooled_prompt_embeds=pooled_prompt_embeds.clone(),
                        text_ids=text_ids.clone(),
                        latent_image_ids=latent_image_ids.clone()
                        if latent_image_ids is not None
                        else None,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        batch_size=latents.shape[0],
                        timesteps=timesteps.clone(),
                        negative_prompt_embeds=negative_prompt_embeds.clone()
                        if do_true_cfg
                        else None,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.clone()
                        if do_true_cfg
                        else None,
                        negative_text_ids=negative_text_ids.clone()
                        if do_true_cfg
                        else None,
                        seed_list=list(seed_list) if seed_list is not None else None,
                        noise_pred=noise_pred.clone(),
                        timestep_value=t.clone(),
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {
                        k: locals()[k] for k in callback_on_step_end_tensor_inputs
                    }
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        image = self._decode_latents(latents, height, width, output_type)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, captured_cache)
        return FlashBonFluxOutput(images=image, generation_cache=captured_cache)

    def _resume_from_cache(
        self,
        generation_cache: GenerationCache,
        resume_from_step: Optional[int] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        device = self._execution_device

        # 1. Restore state
        latents = generation_cache.latents.to(device)
        cached_noise_pred = generation_cache.noise_pred.to(device)

        if resume_from_step is None:
            resume_from_step = generation_cache.step_index + 1

        # 2. Restore timesteps
        if generation_cache.timesteps is not None:
            timesteps = generation_cache.timesteps.to(device)
        else:
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            timesteps, _ = retrieve_timesteps(
                self.scheduler, generation_cache.num_inference_steps, device, mu=mu
            )

        # 3. Embeds & guidance
        prompt_embeds = generation_cache.prompt_embeds.to(device)
        pooled_prompt_embeds = generation_cache.pooled_prompt_embeds.to(device)
        text_ids = generation_cache.text_ids.to(device)
        latent_image_ids = (
            generation_cache.latent_image_ids.to(device)
            if generation_cache.latent_image_ids is not None
            else None
        )

        guidance_scale = generation_cache.guidance_scale
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        negative_prompt_embeds = generation_cache.negative_prompt_embeds
        negative_pooled_prompt_embeds = generation_cache.negative_pooled_prompt_embeds
        negative_text_ids = generation_cache.negative_text_ids
        do_true_cfg = negative_prompt_embeds is not None
        if do_true_cfg:
            negative_prompt_embeds = negative_prompt_embeds.to(device)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
            negative_text_ids = negative_text_ids.to(device)

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        # 4. Latent adjustment to land on `resume_from_step`
        cached_step_index = generation_cache.step_index
        t_cached = timesteps[cached_step_index]

        if resume_from_step <= cached_step_index:
            # backwards extrapolation: add noise back via inverse Euler step
            t_target = timesteps[resume_from_step]
            dt = t_target - t_cached  # positive (Flux timesteps go 1.0 -> 0.0)
            jump_size = cached_step_index - resume_from_step
            if jump_size > (len(timesteps) * 0.2):
                logger.warning(
                    f"Large backwards jump ({cached_step_index} -> {resume_from_step}); "
                    "may cause saturated/corrupted images due to linear approximation error."
                )
            latents = latents + dt * cached_noise_pred
        elif resume_from_step == cached_step_index + 1:
            # forward bridge: apply the cached prediction to advance exactly one step
            self.scheduler._step_index = cached_step_index
            latents = self.scheduler.step(
                cached_noise_pred, t_cached, latents, return_dict=False
            )[0]

        # 5. Denoising loop from resume_from_step
        self.scheduler._step_index = resume_from_step
        remaining_timesteps = timesteps[resume_from_step:]

        with self.progress_bar(total=len(remaining_timesteps)) as progress_bar:
            for i, t in enumerate(remaining_timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = neg_noise_pred + guidance_scale * (
                        noise_pred - neg_noise_pred
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {
                        k: locals()[k] for k in callback_on_step_end_tensor_inputs
                    }
                    callback_outputs = callback_on_step_end(
                        self, resume_from_step + i, t, callback_kwargs
                    )
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        self._current_timestep = None
        image = self._decode_latents(
            latents, generation_cache.height, generation_cache.width, output_type
        )
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, None)
        return FlashBonFluxOutput(images=image, generation_cache=None)

    # ------------------------------------------------------------------
    def _decode_latents(self, latents, height, width, output_type):
        if output_type == "latent":
            return latents
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type=output_type)
