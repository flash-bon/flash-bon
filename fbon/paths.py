"""Centralized filesystem locations. Every module reads paths from here -- nothing else hardcodes
an absolute path. Override any of these with an environment variable to run on a different machine:

    FBON_CACHE_DIR    shared cache root (model weights + the server conda envs). Falls back to
                      $HF_HOME, then to the default below.
    HF_HOME           HuggingFace weight cache (defaults to CACHE_DIR).
    FBON_VLLM_ENV     conda env (name or prefix path) for the vLLM verifier server.
    FBON_VQA_ENV      conda env (name or prefix path) for the VQAScore server.
    FBON_CONDA_SH     path to conda's profile.d/conda.sh (to `conda activate` inside the servers).
    FBON_CONDA_EXE    the conda executable (used to locate conda.sh if FBON_CONDA_SH is unset).
"""

import os
from pathlib import Path

# This package (<repo>/fbon) and the repo root (<repo>).
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent

# Shared cache root holding the model weights and the server conda envs.
CACHE_DIR = (
    os.environ.get("FBON_CACHE_DIR")
    or os.environ.get("HF_HOME")
    or "/fs/nexus-projects/mt_sec/cache"
)

# HuggingFace weight cache (defaults to CACHE_DIR).
HF_HOME = os.environ.get("HF_HOME", CACHE_DIR)

# Conda envs the servers run in. Each server is launched with `conda activate <env>` so the env's
# PATH / LD_LIBRARY_PATH / console tools (e.g. ffmpeg) are set up. Values may be a conda env NAME or
# a full prefix PATH; defaults are prefixes under CACHE_DIR.
VLLM_ENV = os.environ.get("FBON_VLLM_ENV", os.path.join(CACHE_DIR, "flashbon_vllm"))
VQA_ENV = os.environ.get("FBON_VQA_ENV", os.path.join(CACHE_DIR, "flashbon_vqa"))

# conda.sh, needed to `conda activate` inside a launched subprocess. Derived from the conda
# executable (which conda sets as $CONDA_EXE when a base/env is active) unless given explicitly.
_CONDA_EXE = os.environ.get("FBON_CONDA_EXE") or os.environ.get("CONDA_EXE", "")


def _default_conda_sh() -> str:
    if _CONDA_EXE:
        cand = os.path.join(os.path.dirname(os.path.dirname(_CONDA_EXE)), "etc", "profile.d", "conda.sh")
        if os.path.exists(cand):
            return cand
    return ""


CONDA_SH = os.environ.get("FBON_CONDA_SH") or _default_conda_sh()
