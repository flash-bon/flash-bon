"""VQAScore final-eval metric (HTTP client).

POSTs ``{"image_path", "prompt"}`` to the VQAScore server's ``/infer`` and reads ``{"score","time"}``.
The server reads ``image_path`` off the shared filesystem. Returns, per image, a dict with
``final_candidate_score`` and ``final_candidate_time``.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests


class VqascoreMetric:
    name = "vqascore"

    def __init__(self, endpoint: str = "http://127.0.0.1:5005", timeout: float = 120.0, max_workers: int = 8):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.max_workers = max_workers
        self._session = requests.Session()

    def check_health(self):
        status = self._session.get(f"{self.endpoint}/health", timeout=10).json().get("status")
        if status != "ok":
            raise RuntimeError(f"vqascore server not ready ({status}) at {self.endpoint}")

    def _score_one(self, image_path: str, prompt: str) -> Dict[str, Any]:
        t0 = time.time()
        r = self._session.post(f"{self.endpoint}/infer",
                               json={"image_path": image_path, "prompt": prompt}, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()
        return {"final_candidate_score": float(out["score"]),
                "final_candidate_time": float(out.get("time", time.time() - t0))}

    def score(self, image_paths: List[str], prompt: str) -> List[Optional[Dict[str, Any]]]:
        results: List[Optional[Dict[str, Any]]] = [None] * len(image_paths)

        def work(i):
            try:
                return i, self._score_one(image_paths[i], prompt)
            except Exception as e:
                return i, {"final_candidate_score": None, "final_candidate_error": str(e)}

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for i, res in ex.map(work, range(len(image_paths))):
                results[i] = res
        return results
