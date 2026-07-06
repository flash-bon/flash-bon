"""Pairwise-tournament verifier for in-loop candidate selection.

Wraps the multi-stage ranking tournament so the budget loop can drive selection. Unlike a
pointwise scorer, the tournament keeps state ACROSS batches for a prompt:

  * ``reset(prompt)`` starts a fresh tournament at the beginning of each prompt.
  * ``select(...)`` is given a batch's saved image paths + seeds; it runs the tournament for that
    batch (reading files, comparing them via the vLLM server), updates state, and returns the
    within-batch ranking (best -> worst) plus per-seed ELO scores.
  * ``global_scores_by_seed()`` exposes the running global ELO across all batches, so the finalize
    step refines the true global top-k (later cross-batch comparisons can re-rank earlier seeds).
"""

import os
from typing import Dict, List, Optional

from PIL import Image

from .ranking import MultiStageFilterStrategy, check_vllm_health


class PairwiseRankingVerifier:
    kind = "pairwise"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        cross_batch_top_k: int = 3,
        max_workers: int = 4,
        timeout: float = 60.0,
    ):
        endpoint = endpoint or os.getenv("VLLM_ENDPOINT")
        if not endpoint:
            raise ValueError("pairwise verifier needs a vLLM endpoint (endpoint=/$VLLM_ENDPOINT).")
        self.endpoint = endpoint
        self.model = model or "Qwen/Qwen2.5-VL-7B-Instruct"
        self._kw = dict(max_workers=max_workers, timeout=timeout,
                        cross_batch_top_k=cross_batch_top_k, enable_cross_batch=True)
        self._strategy: Optional[MultiStageFilterStrategy] = None
        self._prompt: Optional[str] = None
        self._batch_counter = 0

    def check_health(self, retries: int = 3):
        check_vllm_health(self.endpoint, self.model, retries=retries)

    def reset(self, prompt: str):
        self._strategy = MultiStageFilterStrategy(self.endpoint, self.model, prompt, **self._kw)
        self._prompt = prompt
        self._batch_counter = 0

    def select(self, images: List[Image.Image], prompt: str, image_paths: List[str],
               seeds: List[int], batch_id: Optional[int] = None) -> dict:
        """Run the tournament on one batch; return within-batch ranking + per-seed ELO scores."""
        if self._strategy is None or self._prompt != prompt:
            self.reset(prompt)
        bid = batch_id if batch_id is not None else self._batch_counter
        self._batch_counter += 1

        seed_strs = [str(s) for s in seeds]
        self._strategy.add_batch(image_paths, batch_id=bid, seeds=seed_strs)

        # current global results keyed by seed
        all_res = {str(r["seed"]): r for r in self._strategy.get_all_results()}
        order = sorted(range(len(seed_strs)),
                       key=lambda i: all_res.get(seed_strs[i], {}).get("rank", 1e9))
        scores = [float(all_res.get(s, {}).get("elo_rating", 1500.0)) for s in seed_strs]
        return {"ranking": order, "scores": scores,
                "stats": {"vlm_calls": self._strategy.state.total_vlm_calls}}

    def global_scores_by_seed(self) -> Dict[int, float]:
        """Running global ELO per (integer) seed across every batch seen so far."""
        out = {}
        for r in self._strategy.get_all_results():
            s = str(r["seed"])
            if s.lstrip("-").isdigit():
                out[int(s)] = float(r.get("elo_rating", 0.0))
        return out

    @property
    def total_vlm_calls(self) -> int:
        return self._strategy.state.total_vlm_calls if self._strategy else 0
