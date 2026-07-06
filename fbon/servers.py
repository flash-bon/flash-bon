"""Auto-launch (or reuse) the two API servers the pipeline talks to.

* ``ensure_vllm``     -- OpenAI-compatible vLLM server hosting the Qwen-VL verifier model.
* ``ensure_vqascore`` -- Flask VQAScore server (final-eval metric).
"""

import os
import shlex
import subprocess
import time
from urllib.parse import urlparse

import requests

from .paths import CONDA_SH, HF_HOME, PACKAGE_DIR, VLLM_ENV, VQA_ENV

_SERVE_VQASCORE = os.path.join(PACKAGE_DIR, "serve_vqascore.py")

DEFAULT_VLLM_ENDPOINT = "http://127.0.0.1:8100"
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_VQA_ENDPOINT = "http://127.0.0.1:5005"
DEFAULT_VQA_MODEL = "qwen2.5-vl-7b"


def _server_env(gpu):
    """Env for a launched server: pin its GPU and set HF_HOME. PATH / library setup comes from
    activating the server's conda env (see _launch_in_env)."""
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=HF_HOME)


def _launch_in_env(conda_env, cmd, gpu, log_path):
    """Launch ``cmd`` (a list, e.g. ["python", ...]) as a DETACHED server inside ``conda_env`` (a
    conda env NAME or prefix PATH). The env is ``conda activate``d first so its PATH / LD_LIBRARY_PATH
    / console tools (ffmpeg, ...) apply; ``exec`` then replaces the shell with the server. Returns the
    Popen handle."""
    if not CONDA_SH:
        raise RuntimeError(
            "cannot locate conda.sh to activate the server env; run from a conda shell or set "
            "FBON_CONDA_SH / FBON_CONDA_EXE (see fbon/paths.py)."
        )
    inner = " ".join(shlex.quote(a) for a in cmd)
    script = f"source {shlex.quote(CONDA_SH)} && conda activate {shlex.quote(conda_env)} && exec {inner}"
    logf = open(log_path, "a")
    return subprocess.Popen(
        ["bash", "-c", script],
        env=_server_env(gpu),
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _get(url, timeout=3):
    try:
        return requests.get(url, timeout=timeout)
    except requests.exceptions.RequestException:
        return None


def ensure_vllm(
    endpoint=DEFAULT_VLLM_ENDPOINT,
    gpu=1,
    model=DEFAULT_VLLM_MODEL,
    timeout=900,
    log_dir="logs",
    max_model_len=4096,
    gpu_memory_utilization=0.55,
):
    """Guarantee a vLLM server is reachable at ``endpoint``; launch one (detached) if not.

    ``gpu_memory_utilization`` defaults below 1 so a VQAScore model can share the same GPU.
    """
    endpoint = endpoint or DEFAULT_VLLM_ENDPOINT
    r = _get(endpoint.rstrip("/") + "/v1/models")
    if r is not None and r.ok:
        print(f"[vllm] reusing server already up at {endpoint}", flush=True)
        return endpoint

    port = urlparse(endpoint).port or 8100
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vllm_{port}.log")
    print(
        f"[vllm] launching {model} in env {VLLM_ENV} on GPU {gpu} (port {port}); log: {log_path}",
        flush=True,
    )
    proc = _launch_in_env(
        VLLM_ENV,
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(max_model_len),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--load-format",
            "auto",
            "--enforce-eager",
        ],
        gpu,
        log_path,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        r = _get(endpoint.rstrip("/") + "/v1/models")
        if r is not None and r.ok:
            print("[vllm] ready", flush=True)
            return endpoint
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM exited early (code {proc.returncode}); see {log_path}"
            )
        time.sleep(5)
    raise RuntimeError(f"vLLM not ready within {timeout}s; see {log_path}")


def ensure_vqascore(
    endpoint=DEFAULT_VQA_ENDPOINT,
    gpu=1,
    model=DEFAULT_VQA_MODEL,
    timeout=900,
    log_dir="logs",
):
    """Guarantee a VQAScore server is reachable at ``endpoint``; launch one (detached) if not."""
    endpoint = endpoint or DEFAULT_VQA_ENDPOINT
    base = endpoint.rstrip("/")
    port = urlparse(endpoint).port or 5005
    r = _get(base + "/health")
    if r is not None and r.status_code == 200 and r.json().get("status") == "ok":
        print(f"[vqascore] reusing server already up at {endpoint}", flush=True)
        return endpoint
    if (
        r is not None
    ):  # a stale/errored server holds the port -> free it before relaunching
        print(
            f"[vqascore] stale server on port {port} (status {r.status_code}); restarting it",
            flush=True,
        )
        subprocess.run(
            ["pkill", "-f", f"serve_vqascore.py .*--port {port}"], check=False
        )
        time.sleep(2)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqascore_{port}.log")
    print(
        f"[vqascore] launching {model} in env {VQA_ENV} on GPU {gpu} (port {port}); log: {log_path}",
        flush=True,
    )
    proc = _launch_in_env(
        VQA_ENV,
        [
            "python",
            _SERVE_VQASCORE,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--vqascore_model",
            model,
        ],
        gpu,
        log_path,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        r = _get(base + "/health")
        if r is not None and r.status_code == 200 and r.json().get("status") == "ok":
            print("[vqascore] ready", flush=True)
            return endpoint
        if r is not None and r.status_code == 500:
            raise RuntimeError(f"vqascore server load error; see {log_path}: {r.text}")
        if proc.poll() is not None:
            raise RuntimeError(
                f"vqascore exited early (code {proc.returncode}); see {log_path}"
            )
        time.sleep(5)
    raise RuntimeError(f"vqascore not ready within {timeout}s; see {log_path}")
