"""Model registry: maps a model key to its pipeline + sensible defaults."""

from dataclasses import dataclass
from typing import Type

import torch

from .flux import FlashBonFluxPipeline
from .wan import FlashBonWanPipeline


@dataclass(frozen=True)
class ModelSpec:
    key: str
    pipeline_cls: Type
    repo: str
    height: int
    width: int
    guidance_scale: float
    steps: int
    task: str  # short label used in the output tree
    config: str  # default caching config (relative to repo root)


MODEL_REGISTRY = {
    "flux-dev": ModelSpec(
        key="flux-dev",
        pipeline_cls=FlashBonFluxPipeline,
        repo="black-forest-labs/FLUX.1-dev",
        height=1024,
        width=1024,
        guidance_scale=3.5,
        steps=50,
        task="flux-dev",
        config="configs/flux_dev.json",
    ),
    "wan-1.3b": ModelSpec(
        key="wan-1.3b",
        pipeline_cls=FlashBonWanPipeline,
        repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        height=480,
        width=832,
        guidance_scale=5.0,
        steps=50,
        task="t2i-1.3B",
        config="configs/wan_1_3b.json",
    ),
    "wan-14b": ModelSpec(
        key="wan-14b",
        pipeline_cls=FlashBonWanPipeline,
        repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        height=480,
        width=832,
        guidance_scale=5.0,
        steps=50,
        task="t2i-14B",
        config="configs/wan_14b.json",
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{key}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


def _set_attention_backend(pipe, backend: str):
    if backend in (None, "default"):
        return "default"
    transformer = getattr(pipe, "transformer", None)
    if transformer is None or not hasattr(transformer, "set_attention_backend"):
        print(
            "[load] transformer has no set_attention_backend; using default attention"
        )
        return "default"
    # diffusers backend names: "flash" (flash_attn FA2), "native" (SDPA).
    name = {"flash": "flash", "sdpa": "native", "native": "native"}.get(
        backend, backend
    )
    try:
        transformer.set_attention_backend(name)
        print(f"[load] attention backend = {name}")
        return name
    except Exception as e:  # missing flash_attn, unsupported version, etc.
        print(f"[load] attention backend '{name}' unavailable, using default SDPA: {e}")
        return "default"


def _iter_blocks(transformer):
    for attr in ("blocks", "transformer_blocks", "single_transformer_blocks"):
        block_list = getattr(transformer, attr, None)
        if block_list is not None:
            for blk in block_list:
                yield blk


def _compile_transformer_blocks(transformer):
    n = 0
    for blk in _iter_blocks(transformer):
        if not hasattr(blk, "_fb_orig_forward"):
            blk._fb_orig_forward = blk.forward
            blk._fb_compiled_forward = torch.compile(blk.forward)
            blk.forward = blk._fb_compiled_forward
        n += 1
    transformer._fb_block_compile_ready = True
    print(f"[load] toggleable block-level compile prepared on {n} transformer blocks")


def set_blocks_compiled(transformer, on: bool):
    if not getattr(transformer, "_fb_block_compile_ready", False):
        return
    for blk in _iter_blocks(transformer):
        if hasattr(blk, "_fb_orig_forward"):
            blk.forward = blk._fb_compiled_forward if on else blk._fb_orig_forward


def load_pipeline(
    key: str,
    dtype=torch.bfloat16,
    device="cuda",
    attn_backend="flash",
    compile=False,
    compile_blocks=False,
):
    spec = get_model_spec(key)
    pipe = spec.pipeline_cls.from_pretrained(spec.repo, torch_dtype=dtype).to(device)
    _set_attention_backend(pipe, attn_backend)
    if compile:
        torch.set_float32_matmul_precision(
            "high"
        )  # TF32 for stray fp32 matmuls (RoPE/norms)
        pipe.transformer.compile()  # in-place: keeps module identity
        print(
            "[load] transformer.compile() enabled (first call compiles; warmup absorbs it)"
        )
    elif compile_blocks:
        torch.set_float32_matmul_precision("high")
        _compile_transformer_blocks(pipe.transformer)
    return pipe, spec
