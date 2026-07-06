"""End-to-end budget-constrained generation.

Given a text prompt and a wall-clock budget, this:

  1. Repeatedly drafts batches of cheap candidates (caching + layer/timestep skipping), capturing
     each draft's state so it can be finished later.
  2. Selects across all drafts with a pairwise VLM tournament (stateful across batches).
  3. Refines the global top-k to full quality in one batched resume from their captured states.
  4. Scores the refined top-k with VQAScore.

Pass ``--method bon`` for plain best-of-N: candidates are generated at full quality (no caching or
skipping) and selected by the same tournament, with no draft/refine (steps 1 and 3 collapse).

The vLLM verifier server and the VQAScore server are launched automatically (each on its own GPU)
and reused across runs. Diffusion runs on the GPU the process was given (e.g. CUDA_VISIBLE_DEVICES=0).
Filesystem locations (weight cache, server conda envs) are configured in ``fbon/paths.py`` and can
be overridden with the ``FBON_CACHE_DIR`` / ``HF_HOME`` / ``FBON_VLLM_ENV`` / ``FBON_VQA_ENV``
environment variables.

Example:
    CUDA_VISIBLE_DEVICES=0 python run.py --prompt "a photo of four giraffes" --budget_seconds 150 --model_key wan-1.3b
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Make the local `fbon` package importable regardless of the current working directory, and take
# priority over any same-named package that might be installed in the environment.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from PIL import Image

from fbon.caching import (
    apply_lazy_bon_scaling,
    remove_lazy_bon_scaling,
    reset_lazy_bon_scaling,
)
from fbon.cache_state import merge_caches, slice_cache
from fbon.config import load_lazybon_config
from fbon.models import load_pipeline, set_blocks_compiled
from fbon.servers import ensure_vllm, ensure_vqascore
from fbon.verifier import PairwiseRankingVerifier
from fbon.vqascore import VqascoreMetric

_REPO = Path(__file__).resolve().parent


def _sync(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _slug(text: str, n: int = 20) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in text.lower())[:n].strip("_")
    return f"{keep}_{hashlib.md5(text.encode()).hexdigest()[:10]}"


def draft_batch(pipe, config, prompt, seeds, cap, args):
    """Generate one batch. flash-bon: cheap cached drafts + captured state (finished later). bon:
    full-quality images with no caching/skipping and no capture. Returns (images, cache_or_None, s)."""
    remove_lazy_bon_scaling(pipe.transformer)
    if args.method == "flash-bon":
        apply_lazy_bon_scaling(pipe.transformer, config)
        reset_lazy_bon_scaling(pipe.transformer)
    t0 = time.perf_counter()
    out = pipe(
        prompt=[prompt] * len(seeds),
        seed_list=seeds,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        capture_at_step=(cap if args.method == "flash-bon" else None),
    )
    _sync(pipe._execution_device)
    if args.method == "flash-bon" and out.generation_cache is None:
        raise RuntimeError("No generation_cache captured; check capture_step < steps.")
    return out.images, out.generation_cache, time.perf_counter() - t0


def refine_from_caches(pipe, caches, rfrom):
    """Resume a list of single-item draft caches to full quality in ONE batched pipe() call.

    The refine runs with block-compile toggled OFF (a new batch shape trips a torch.compile RoPE bug
    on FLUX; and one-off refine shapes aren't worth recompiling), then compile is restored for the
    next draft. Both toggles are no-ops when block-compile was not prepared."""
    remove_lazy_bon_scaling(pipe.transformer)
    set_blocks_compiled(pipe.transformer, False)
    out = pipe(generation_cache=merge_caches(caches), resume_from_step=rfrom)
    _sync(pipe._execution_device)
    set_blocks_compiled(pipe.transformer, True)
    return out.images


def verify_batch(pipe, verifier, prompt, images, seeds, tmp_dir, batch_id, args):
    """Save drafts, run the pairwise tournament on the batch. Returns (per-seed scores, seconds)."""
    paths = []
    for img, seed in zip(images, seeds):
        p = tmp_dir / f"cand_{seed}.png"
        img.save(p)
        paths.append(str(p))
    t0 = time.perf_counter()
    sel = verifier.select(
        images, prompt, image_paths=paths, seeds=seeds, batch_id=batch_id
    )
    _sync(pipe._execution_device)
    return dict(zip(seeds, sel["scores"])), time.perf_counter() - t0


def run(pipe, spec, config, verifier, metric, prompt, args, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_drafts"
    tmp_dir.mkdir(exist_ok=True)
    final_dir = out_dir / "final_topk"
    final_dir.mkdir(exist_ok=True)

    cap = rfrom = None
    if args.method == "flash-bon":
        cap = config.resume_from_step - 1
        rfrom = cap + 1

    verifier.reset(prompt)

    # ---- warmup: estimate one round (draft+verify) and the refine, un-budgeted ----
    # Warmup drafts are throwaway (only used to warm CUDA/compile and time a round), so write their
    # images to a scratch temp dir instead of the run's _drafts/ -- keeps _drafts to real candidates.
    print(f"[estimate] warmup x{max(1, args.warmup)} ...", flush=True)
    warmup_dir = Path(tempfile.mkdtemp(prefix="fbon_warmup_"))
    t_round, t_refine = 0.0, 0.0
    wseed = args.seed_start + 9_000_000
    for w in range(max(1, args.warmup)):
        seeds = list(range(wseed, wseed + args.batch_size))
        wseed += args.batch_size
        imgs, cache, gt = draft_batch(pipe, config, prompt, seeds, cap, args)
        scores, et = verify_batch(
            pipe, verifier, prompt, imgs, seeds, warmup_dir, None, args
        )
        t_round = max(t_round, gt + et)
        if (
            args.method == "flash-bon" and w == max(1, args.warmup) - 1
        ):  # estimate refine exactly as finalize does (batched)
            order = sorted(seeds, key=lambda s: scores[s], reverse=True)[: args.top_k]
            idx = {s: i for i, s in enumerate(seeds)}
            t0 = time.perf_counter()
            refine_from_caches(
                pipe, [slice_cache(cache, [idx[s]]) for s in order], rfrom
            )
            _sync(pipe._execution_device)
            t_refine = time.perf_counter() - t0
    verifier.reset(prompt)  # wipe warmup tournament state
    remove_lazy_bon_scaling(pipe.transformer)
    print(
        f"[estimate] t_round~{t_round:.2f}s  t_refine(top{args.top_k})~{t_refine:.2f}s",
        flush=True,
    )

    # ---- budget loop: draft + verify batches until the reserve for the refine is hit ----
    scores_by_seed = {}
    cache_by_seed = {}  # single-item draft caches retained for the current top-k (+buffer)
    seed_cursor = args.seed_start
    cum = 0.0
    rnd = 0
    first_batch_s = None

    while rnd < args.max_rounds:
        if rnd > 0 and cum + t_round + t_refine > args.budget_seconds:
            break
        seeds = [seed_cursor + i for i in range(args.batch_size)]
        seed_cursor += args.batch_size
        imgs, cache, gt = draft_batch(pipe, config, prompt, seeds, cap, args)
        scores, et = verify_batch(
            pipe, verifier, prompt, imgs, seeds, tmp_dir, rnd, args
        )
        for s in seeds:
            scores_by_seed[s] = float(scores[s])

        if args.method == "flash-bon":
            keep = set(
                sorted(scores_by_seed, key=lambda s: scores_by_seed[s], reverse=True)[
                    : args.top_k + args.cache_buffer
                ]
            )
            for li, seed in enumerate(seeds):
                if seed in keep and seed not in cache_by_seed:
                    cache_by_seed[seed] = slice_cache(cache, [li])
            for s in [s for s in cache_by_seed if s not in keep]:
                cache_by_seed.pop(s, None)

        cum += gt + et
        if first_batch_s is None:
            first_batch_s = cum
        t_round = max(t_round, gt + et)
        rnd += 1

    search_secs = cum

    gscores = verifier.global_scores_by_seed() or scores_by_seed
    topk_seeds = sorted(gscores, key=lambda s: gscores[s], reverse=True)[: args.top_k]

    refine_start = time.perf_counter()
    finals_by_seed = {}
    mem_seeds, redraft_seeds = [], []
    if args.method == "flash-bon":
        mem_seeds = [s for s in topk_seeds if s in cache_by_seed]
        redraft_seeds = [s for s in topk_seeds if s not in cache_by_seed]
        if mem_seeds:
            images = refine_from_caches(
                pipe, [cache_by_seed[s] for s in mem_seeds], rfrom
            )
            for img, seed in zip(images, mem_seeds):
                finals_by_seed[seed] = img
        for seed in redraft_seeds:
            _, cache, _ = draft_batch(pipe, config, prompt, [seed], cap, args)
            finals_by_seed[seed] = refine_from_caches(pipe, [cache], rfrom)[0]
    refine_secs = time.perf_counter() - refine_start

    # save the finals (refined image for flash-bon; the full-quality candidate itself for bon)
    final_paths = {}
    for rank, seed in enumerate(topk_seeds):
        fp = final_dir / f"rank{rank}_seed{seed}.png"
        if seed in finals_by_seed:
            finals_by_seed[seed].save(fp)
        else:
            shutil.copyfile(tmp_dir / f"cand_{seed}.png", fp)
        final_paths[seed] = str(fp)

    # ---- final eval (VQAScore), un-budgeted ----
    metric_scores = {}
    if metric is not None:
        image_paths = [final_paths[s] for s in topk_seeds]
        per_image = metric.score(image_paths, prompt)
        metric_scores = dict(zip(topk_seeds, per_image))

    # ---- record ----
    rows = []
    for rank, seed in enumerate(topk_seeds):
        row = {
            "rank": rank,
            "seed": seed,
            "verifier_elo": gscores.get(seed),
            "final_image": final_paths[seed],
        }
        row.update(metric_scores.get(seed) or {})
        rows.append(row)
    result = {
        "prompt": prompt,
        "method": args.method,
        "compiled": args.compile,
        "model_key": args.model_key,
        "budget_seconds": args.budget_seconds,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "num_rounds": rnd,
        "num_candidates": len(scores_by_seed),
        "first_batch_output_s": first_batch_s,
        "search_secs": search_secs,
        "refine_secs": refine_secs,
        "total_secs": search_secs + refine_secs,
        "est_t_round": t_round,
        "est_t_refine": t_refine,
        "topk_seeds": topk_seeds,
        "best_seed": topk_seeds[0] if topk_seeds else None,
        "topk_refined_from_cache": len(mem_seeds),
        "topk_redrafted": len(redraft_seeds),
        "total_vlm_calls": verifier.total_vlm_calls,
        "candidates": rows,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    fb = f"{first_batch_s:.1f}" if first_batch_s is not None else "n/a"
    best_vqa = (
        (metric_scores.get(topk_seeds[0]) or {}).get("final_candidate_score")
        if topk_seeds
        else None
    )
    print(
        f"-> [{args.method}] FIRST BATCH @ {fb}s | {rnd} rounds, {len(scores_by_seed)} candidates; "
        f"search {search_secs:.1f}s + refine {refine_secs:.1f}s = {result['total_secs']:.1f}s "
        f"/ budget {args.budget_seconds}s; refined {len(mem_seeds)}/{len(redraft_seeds)} re-drafted; "
        f"best seed {result['best_seed']} vqascore={best_vqa}",
        flush=True,
    )
    print(f"-> results: {out_dir / 'result.json'}", flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--prompt", required=True, help="text prompt to generate for")
    p.add_argument(
        "--method",
        default="flash-bon",
        choices=["flash-bon", "bon"],
        help="flash-bon = cached draft + refine top-k; bon = full-quality best-of-N (no refine)",
    )
    p.add_argument(
        "--budget_seconds",
        type=float,
        default=150.0,
        help="wall-clock budget (draft+verify+refine)",
    )
    p.add_argument(
        "--model_key", default="wan-1.3b", choices=["flux-dev", "wan-1.3b", "wan-14b"]
    )
    p.add_argument(
        "--config",
        default=None,
        help="caching config JSON (default: the model's configs/*.json)",
    )
    p.add_argument(
        "--batch_size", type=int, default=8, help="drafts generated per round"
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="how many candidates to refine to full quality",
    )
    p.add_argument(
        "--cache_buffer",
        type=int,
        default=4,
        help="keep (top_k + this) draft caches so a re-promoted candidate is still cached",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="untimed warmup rounds to estimate round/refine cost",
    )
    p.add_argument("--max_rounds", type=int, default=1000)
    p.add_argument(
        "--steps", type=int, default=None, help="denoising steps (default: the model's)"
    )
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument(
        "--attn_backend", default="flash", choices=["flash", "sdpa", "default"]
    )
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the transformer (ON by default). flash-bon uses toggleable block-level "
        "compile; bon uses top-level compile. Pass --no-compile to disable (e.g. eager A/B).",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="where to write outputs (default: outputs/<model>/<prompt>)",
    )
    # servers
    p.add_argument(
        "--no_final_eval",
        action="store_true",
        help="skip VQAScore (still does draft/select/refine)",
    )
    p.add_argument(
        "--server_gpu", type=int, default=1, help="GPU for the vLLM + VQAScore servers"
    )
    p.add_argument("--vllm_endpoint", default="http://127.0.0.1:8100")
    p.add_argument("--vllm_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--vqa_endpoint", default="http://127.0.0.1:5005")
    p.add_argument("--vqa_model", default="qwen2.5-vl-7b")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- launch/attach the verifier + eval servers (each on its own GPU) ----
    ensure_vllm(endpoint=args.vllm_endpoint, gpu=args.server_gpu, model=args.vllm_model)
    if not args.no_final_eval:
        ensure_vqascore(
            endpoint=args.vqa_endpoint, gpu=args.server_gpu, model=args.vqa_model
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    top_level_compile = args.compile and args.method == "bon"
    block_compile = args.compile and args.method == "flash-bon"
    print(
        f"[load] {args.model_key} on {device} "
        f"(method={args.method}, compile={'block' if block_compile else 'top-level' if top_level_compile else 'off'}) ...",
        flush=True,
    )
    pipe, spec = load_pipeline(
        args.model_key,
        device=device,
        attn_backend=args.attn_backend,
        compile=top_level_compile,
        compile_blocks=block_compile,
    )

    args.steps = args.steps or spec.steps
    args.height = args.height or spec.height
    args.width = args.width or spec.width
    args.guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None else spec.guidance_scale
    )

    config_path = args.config or str(_REPO / spec.config)
    config, base = load_lazybon_config(config_path, sample_steps=args.steps)
    if args.method == "flash-bon" and config is None:
        raise ValueError(
            f"config {config_path} has no optimal_params (baseline); flash-bon needs caching params."
        )
    if args.method == "flash-bon":
        print(
            f"[config] {base.get('config_name')}  resume_from_step={config.resume_from_step}",
            flush=True,
        )

    verifier = PairwiseRankingVerifier(
        endpoint=args.vllm_endpoint, model=args.vllm_model
    )
    verifier.check_health()
    metric = None if args.no_final_eval else VqascoreMetric(endpoint=args.vqa_endpoint)
    if metric is not None:
        metric.check_health()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            _REPO
            / "outputs"
            / args.method
            / args.model_key
            / f"budget{int(args.budget_seconds)}s"
            / _slug(args.prompt)
        )
    )

    run(pipe, spec, config, verifier, metric, args.prompt, args, out_dir)


if __name__ == "__main__":
    main()
