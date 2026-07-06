# Flash-BoN

Budget-constrained, verifier-guided best-of-N generation for diffusion models. Given a prompt and
a wall-clock budget, it drafts cheap candidates (TaylorSeer-style caching + layer/timestep
skipping), selects the best with a pairwise vision-language tournament, refines the top-k to full
quality by resuming denoising from a captured state, and scores them with VQAScore.

## Layout

```
run.py                 end-to-end: prompt + budget -> draft -> select -> refine -> VQAScore
run_examples.sh        copy-paste commands (per method / model / budget)
configs/               per-model caching configs (flux_dev, wan_1_3b, wan_14b)
envs/                  pinned pip requirements for the three conda envs
fbon/
  paths.py             single source of filesystem locations (cache dir, server interpreters)
  models.py            model registry + load_pipeline (+ block/top-level compile)
  flux.py, wan.py      pipeline subclasses with draft-capture / resume
  caching.py           TaylorSeer caching + layer/timestep skipping (diffusers hooks)
  cache_state.py       GenerationCache + batch slice/merge
  config.py            load a caching config JSON
  ranking.py           multi-stage pairwise tournament (ELO across batches)
  verifier.py          tournament wrapper used for in-loop selection
  vqascore.py          VQAScore final-eval client
  servers.py           auto-launch/reuse the vLLM verifier + VQAScore servers
  serve_vqascore.py    the VQAScore HTTP server
```

## Usage

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --prompt "a photo of four giraffes" --budget_seconds 150 --model_key wan-1.3b
```

Outputs are written to `outputs/<method>/<model>/budget<N>s/<prompt>/`:
`result.json` (timings, rankings, VQAScore) and `final_topk/rank<r>_seed<seed>.png`.
See `run_examples.sh` for copy-paste commands.

## Environment setup

Three conda environments (Python 3.12), one per process. `run.py` runs in the **main** env and
launches the other two automatically (each on `--server_gpu`), talking to them over HTTP. Pinned
requirements are in [`envs/`](envs/).

```bash
# 1. main -- diffusion + this repo (run.py runs from source; do NOT pip-install the project)
conda create -n flashbon_env python=3.12 -y && conda activate flashbon_env
pip install -r envs/requirements-main.txt
# optional FlashAttention-2 (needs nvcc + a C++ compiler to build); else run with --attn_backend sdpa
pip install flash-attn==2.8.3 --no-build-isolation

# 2. vLLM verifier -- serves Qwen/Qwen2.5-VL-7B-Instruct
conda create -n flashbon_vllm python=3.12 -y && conda activate flashbon_vllm
pip install -r envs/requirements-vllm.txt

# 3. VQAScore final metric -- t2v_metrics
conda create -n flashbon_vqa python=3.12 -y && conda activate flashbon_vqa
pip install -r envs/requirements-vqa.txt
conda install -c conda-forge ffmpeg -y                # t2v_metrics checks `ffmpeg -version` at import
python -c "import t2v_metrics; print('ok')"           # verify the env is good
```

Each server is launched by `conda activate`-ing its env (`flashbon_vllm` / `flashbon_vqa` under
`$FBON_CACHE_DIR` by default; see [`fbon/paths.py`](fbon/paths.py)). If your envs live elsewhere or
have other names, set `FBON_VLLM_ENV` / `FBON_VQA_ENV` (a conda env name or prefix path); set
`FBON_CONDA_SH` if conda can't be auto-located.