"""Multi-stage pairwise ranking tournament for VLM-based image selection."""

import base64
import math
import mimetypes
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter

_SESSION = requests.Session()
_SESSION.mount("http://", HTTPAdapter(pool_connections=32, pool_maxsize=32))
_SESSION.mount("https://", HTTPAdapter(pool_connections=32, pool_maxsize=32))


# RUBRICS

POINTWISE_RUBRIC = (
    "You are a strict prompt-adherence evaluator. Ignore minor visual quality issues.\n"
    "Judge the image on **only two criteria**:\n"
    " 1. *Object & Scene Match* - Are **all** objects, characters, and key scene elements requested in the prompt present?\n"
    " 2. *Visual Elements Match* - Do colors, styles, compositions, lighting, or other visual attributes appear as described?\n\n"
    "Assign a single integer *Score* (0-10) representing overall prompt adherence:\n"
    "  • 9-10: Perfect semantic match - ALL elements clearly present and correct\n"
    "  • 7-8: Strong match - Most elements correct, minor omissions acceptable\n"
    "  • 5-6: Moderate match - Core concept present but significant gaps\n"
    "  • 3-4: Weak match - Only partial/vague connection to prompt\n"
    "  • 1-2: Minimal match - Barely any connection\n"
    "  • 0: Complete failure - No connection whatsoever\n\n"
    "Finally, output **exactly** three newline-separated lines *with no extra text*:\n"
    "Reasoning: <1-2 sentences>\n"
    "Verdict: <yes|no>\n"
    "Score: <0-10>\n"
)

PAIRWISE_RUBRIC = (
    "You are comparing two AI-generated images (Image A and Image B) against the same text prompt.\n"
    "Your task: Determine which image BETTER matches the prompt.\n\n"
    "Consider these criteria:\n"
    " 1. *Object & Scene Match* - Which image has more of the requested objects/characters/elements?\n"
    " 2. *Visual Elements Match* - Which image better matches described colors, styles, compositions?\n\n"
    "Rules:\n"
    " - Focus ONLY on prompt adherence, not general image quality\n"
    " - If both are equally good/bad, still pick the slightly better one\n"
    " - Only output 'TIE' if they are truly indistinguishable\n\n"
    "Output **exactly** three lines:\n"
    "Reasoning: <1-2 sentences comparing the images>\n"
    "Winner: <A, B, or TIE>\n"
    "Confidence: <high, medium, low>\n"
)


# DATA CLASSES


@dataclass
class ImageCandidate:
    """A single image candidate with its scores and metadata."""

    id: str
    path: Optional[str] = None
    batch_id: Optional[int] = None
    seed: Optional[str] = None

    pointwise_score: Optional[float] = None
    pointwise_reasoning: Optional[str] = None

    elo_rating: float = 1500.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    comparison_count: int = 0
    swiss_score: float = 0.0

    compared_with: Set[str] = field(default_factory=set)
    original_row_data: Dict[str, Any] = field(default_factory=dict)

    def total_comparisons(self) -> int:
        return self.wins + self.losses + self.ties

    def win_rate(self) -> float:
        total = self.total_comparisons()
        if total == 0:
            return 0.5
        return (self.wins + 0.5 * self.ties) / total


@dataclass
class ComparisonResult:
    """Result of a pairwise comparison."""

    id_a: str
    id_b: str
    winner: str  # 'A', 'B', or 'TIE'
    confidence: str  # 'high', 'medium', 'low'
    reasoning: str
    raw_text: str


@dataclass
class RankingState:
    """Persistent state that accumulates across batches."""

    candidates: Dict[str, ImageCandidate] = field(default_factory=dict)
    comparisons: List[ComparisonResult] = field(default_factory=list)
    batch_count: int = 0
    total_vlm_calls: int = 0
    total_eval_time: float = 0.0

    current_best_id: Optional[str] = None
    current_best_score: float = float("-inf")

    comparison_pairs: Set[Tuple[str, str]] = field(default_factory=set)

    def get_rankings(self) -> List[Tuple[str, float]]:
        items = [(c.id, c.elo_rating) for c in self.candidates.values()]
        return sorted(items, key=lambda x: -x[1])

    def get_top_k(self, k: int) -> List[str]:
        return [r[0] for r in self.get_rankings()[:k]]

    def has_compared(self, id_a: str, id_b: str) -> bool:
        return tuple(sorted([id_a, id_b])) in self.comparison_pairs

    def record_comparison(self, id_a: str, id_b: str):
        self.comparison_pairs.add(tuple(sorted([id_a, id_b])))
        if id_a in self.candidates:
            self.candidates[id_a].compared_with.add(id_b)
        if id_b in self.candidates:
            self.candidates[id_b].compared_with.add(id_a)

    def update_best(self):
        rankings = self.get_rankings()
        if rankings:
            self.current_best_id = rankings[0][0]
            self.current_best_score = rankings[0][1]


# PARSING HELPERS


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+$", "", s, flags=re.M)


def parse_pointwise_output(raw_text: str) -> Dict[str, Any]:
    text = _normalize_text(raw_text)
    reasoning_match = re.search(
        r"Reason(?:ing)?\s*[:=\-]\s*(.+?)(?=\n|Verdict|$)", text, re.I | re.S
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    score_match = re.search(r"Score\s*[:=\-]\s*(\d+(?:\.\d+)?)", text, re.I)
    score = None
    if score_match:
        try:
            score = max(0.0, min(10.0, float(score_match.group(1))))
        except ValueError:
            pass
    return {"score": score, "reasoning": reasoning, "raw_text": raw_text}


def parse_pairwise_output(raw_text: str) -> Dict[str, Any]:
    text = _normalize_text(raw_text)
    reasoning_match = re.search(
        r"Reason(?:ing)?\s*[:=\-]\s*(.+?)(?=\n|Winner|$)", text, re.I | re.S
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    winner_match = re.search(
        r"Winner\s*[:=\-]\s*(A|B|TIE|Image\s*A|Image\s*B)", text, re.I
    )
    winner = None
    if winner_match:
        w = winner_match.group(1).upper().strip()
        winner = "A" if "A" in w else ("B" if "B" in w else "TIE")
    conf_match = re.search(r"Confidence\s*[:=\-]\s*(high|medium|low)", text, re.I)
    confidence = conf_match.group(1).lower() if conf_match else "medium"
    return {
        "winner": winner,
        "confidence": confidence,
        "reasoning": reasoning,
        "raw_text": raw_text,
    }


def extract_seed_from_path(path: str) -> str:
    try:
        filename = os.path.basename(path)
        if "seed" in filename:
            return filename.split("_")[0].replace("seed", "")
    except (AttributeError, IndexError):
        pass
    return str(hash(path) % 1000000)


# VLM INTERFACE


def _img_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _vllm_chat(
    endpoint: str, model: str, messages, max_tokens=256, temperature=0.2, timeout=60.0
) -> str:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = _SESSION.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def pointwise_evaluate(
    image_path: str, prompt: str, endpoint: str, model: str, timeout: float = 60.0
) -> Dict[str, Any]:
    """Evaluate a single image with pointwise scoring."""
    b64 = _img_to_b64(image_path)
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    judge_query = (
        f"{POINTWISE_RUBRIC}\nUser prompt:\n```{prompt}```\n\n"
        "Task: Evaluate the image and respond as instructed above."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                },
                {"type": "text", "text": judge_query},
            ],
        }
    ]
    return parse_pointwise_output(
        _vllm_chat(endpoint, model, messages, timeout=timeout)
    )


def pairwise_compare(
    image_path_a: str,
    image_path_b: str,
    prompt: str,
    endpoint: str,
    model: str,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Compare two images and determine which better matches the prompt."""
    b64_a = _img_to_b64(image_path_a)
    b64_b = _img_to_b64(image_path_b)
    mime_a = mimetypes.guess_type(image_path_a)[0] or "image/png"
    mime_b = mimetypes.guess_type(image_path_b)[0] or "image/png"
    judge_query = (
        f"{PAIRWISE_RUBRIC}\nUser prompt:\n```{prompt}```\n\n"
        "Task: Compare Image A and Image B, then respond as instructed above."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image A:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_a};base64,{b64_a}"},
                },
                {"type": "text", "text": "Image B:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_b};base64,{b64_b}"},
                },
                {"type": "text", "text": judge_query},
            ],
        }
    ]
    return parse_pairwise_output(
        _vllm_chat(endpoint, model, messages, max_tokens=256, timeout=timeout)
    )


def check_vllm_health(
    endpoint: str, model: str, timeout: float = 60.0, retries: int = 3
) -> None:
    """Check that the vLLM server is reachable and responding (first call may be slow)."""
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    probe = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
        "max_tokens": 1,
    }
    for attempt in range(1, retries + 1):
        try:
            requests.post(url, json=probe, timeout=timeout).raise_for_status()
            return
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"vLLM health check failed for {endpoint}: {e}")
            print(f"  Health check attempt {attempt}/{retries} failed, retrying...")


# ELO


class ELOSystem:
    """ELO rating system for pairwise comparisons."""

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(
        self, rating_a: float, rating_b: float, winner: str, confidence: str = "medium"
    ) -> Tuple[float, float]:
        k = self.k_factor * {"high": 1.2, "medium": 1.0, "low": 0.7}.get(
            confidence, 1.0
        )
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        if winner == "A":
            actual_a, actual_b = 1.0, 0.0
        elif winner == "B":
            actual_a, actual_b = 0.0, 1.0
        else:
            actual_a, actual_b = 0.5, 0.5
        return rating_a + k * (actual_a - expected_a), rating_b + k * (
            actual_b - expected_b
        )


# TOURNAMENT


class MultiStageFilterStrategy:
    """Progressive multi-stage filtering tournament, stateful across batches.

    Per batch: (1) pointwise-score all images, keep the top ``stage1_keep_ratio``; (2) sparse
    pairwise among survivors, keep the top ``stage2_keep_ratio``; (3) dense pairwise among the
    finalists; (4) a few cross-batch comparisons of this batch's winner against the running
    leaderbord. All comparisons update global ELO ratings.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        prompt: str,
        timeout: float = 60.0,
        max_workers: int = 4,
        cross_batch_top_k: int = 3,
        enable_cross_batch: bool = True,
        stage1_keep_ratio: float = 0.5,
        stage2_keep_ratio: float = 0.5,
        stage2_comparisons_per_item: float = 1.5,
    ):
        self.endpoint = endpoint
        self.model = model
        self.prompt = prompt
        self.timeout = timeout
        self.max_workers = max_workers
        self.cross_batch_top_k = cross_batch_top_k
        self.enable_cross_batch = enable_cross_batch
        self.stage1_keep_ratio = stage1_keep_ratio
        self.stage2_keep_ratio = stage2_keep_ratio
        self.stage2_comparisons_per_item = stage2_comparisons_per_item
        self.state = RankingState()
        self.elo = ELOSystem()

    # ---- accessors ----
    def get_rankings(self) -> List[Tuple[str, float]]:
        return self.state.get_rankings()

    def get_top_k(self, k: int) -> List[str]:
        return self.state.get_top_k(k)

    def get_best(self) -> Optional[str]:
        top = self.get_top_k(1)
        return top[0] if top else None

    def get_best_path(self) -> Optional[str]:
        best_id = self.get_best()
        if best_id and best_id in self.state.candidates:
            return self.state.candidates[best_id].path
        return None

    def get_best_seed(self) -> Optional[str]:
        best_id = self.get_best()
        if best_id and best_id in self.state.candidates:
            return self.state.candidates[best_id].seed
        return None

    def get_all_results(self) -> List[Dict[str, Any]]:
        rankings = self.get_rankings()
        rank_map = {cid: rank for rank, (cid, _) in enumerate(rankings)}
        results = []
        for cid, cand in self.state.candidates.items():
            result = {
                "seed": cand.seed,
                "candidate_id": cid,
                "batch_id": cand.batch_id,
                "path": cand.path,
                "elo_rating": cand.elo_rating,
                "rank": rank_map.get(cid, -1),
                "pointwise_score": cand.pointwise_score,
                "wins": cand.wins,
                "losses": cand.losses,
                "ties": cand.ties,
                "comparison_count": cand.comparison_count,
                "swiss_score": cand.swiss_score,
            }
            result.update(cand.original_row_data)
            results.append(result)
        return results

    # ---- internals ----
    def _create_candidate(
        self, path: str, batch_id: int, idx: int, seed: Optional[str] = None
    ) -> ImageCandidate:
        if seed is None:
            seed = extract_seed_from_path(path)
        cid = f"batch{batch_id}_img{idx}_seed{seed}"
        return ImageCandidate(id=cid, path=path, batch_id=batch_id, seed=str(seed))

    def _do_comparison(self, id_a: str, id_b: str) -> ComparisonResult:
        result = pairwise_compare(
            self.state.candidates[id_a].path,
            self.state.candidates[id_b].path,
            self.prompt,
            self.endpoint,
            self.model,
            self.timeout,
        )
        return ComparisonResult(
            id_a=id_a,
            id_b=id_b,
            winner=result.get("winner", "TIE"),
            confidence=result.get("confidence", "medium"),
            reasoning=result.get("reasoning", ""),
            raw_text=result.get("raw_text", ""),
        )

    def _update_from_comparison(self, result: ComparisonResult):
        cand_a = self.state.candidates[result.id_a]
        cand_b = self.state.candidates[result.id_b]
        cand_a.elo_rating, cand_b.elo_rating = self.elo.update_ratings(
            cand_a.elo_rating, cand_b.elo_rating, result.winner, result.confidence
        )
        cand_a.comparison_count += 1
        cand_b.comparison_count += 1
        if result.winner == "A":
            cand_a.wins += 1
            cand_b.losses += 1
        elif result.winner == "B":
            cand_b.wins += 1
            cand_a.losses += 1
        else:
            cand_a.ties += 1
            cand_b.ties += 1
        self.state.comparisons.append(result)
        self.state.record_comparison(result.id_a, result.id_b)

    def _execute_comparisons_parallel(
        self, pairs: List[Tuple[str, str]]
    ) -> List[ComparisonResult]:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._do_comparison, a, b) for a, b in pairs]
            for future in as_completed(futures):
                results.append(future.result())
        return results

    # ---- main entry ----
    def add_batch(
        self,
        image_paths: List[str],
        batch_id: Optional[int] = None,
        seeds: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if batch_id is None:
            batch_id = self.state.batch_count

        t0 = time.time()
        vlm_calls = 0

        # Stage 1: pointwise screen
        pointwise_results = []

        def evaluate_one(idx, path):
            return (
                idx,
                path,
                pointwise_evaluate(
                    path, self.prompt, self.endpoint, self.model, self.timeout
                ),
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(evaluate_one, i, p) for i, p in enumerate(image_paths)
            ]
            for future in as_completed(futures):
                pointwise_results.append(future.result())
        vlm_calls += len(image_paths)

        all_ids = []
        for idx, path, result in sorted(pointwise_results, key=lambda x: x[0]):
            seed = seeds[idx] if seeds else None
            candidate = self._create_candidate(path, batch_id, idx, seed=seed)
            candidate.pointwise_score = result.get("score")
            candidate.elo_rating = 1500 + ((result.get("score") or 5) - 5) * 100
            self.state.candidates[candidate.id] = candidate
            all_ids.append(candidate.id)

        sorted_by_pointwise = sorted(
            all_ids, key=lambda x: -(self.state.candidates[x].pointwise_score or 0)
        )
        stage1_survivors = sorted_by_pointwise[
            : max(2, int(len(all_ids) * self.stage1_keep_ratio))
        ]

        # Stage 2: sparse pairwise
        stage2_pairs_count = int(
            len(stage1_survivors) * self.stage2_comparisons_per_item
        )
        sorted_survivors = sorted(
            stage1_survivors, key=lambda x: -self.state.candidates[x].elo_rating
        )
        stage2_pairs = []
        for i in range(min(stage2_pairs_count, len(sorted_survivors) - 1)):
            idx = i % (len(sorted_survivors) - 1)
            pair = (sorted_survivors[idx], sorted_survivors[idx + 1])
            if not self.state.has_compared(pair[0], pair[1]):
                stage2_pairs.append(pair)

        if stage2_pairs:
            for result in self._execute_comparisons_parallel(stage2_pairs):
                self._update_from_comparison(result)
            vlm_calls += len(stage2_pairs)

        sorted_survivors = sorted(
            stage1_survivors, key=lambda x: -self.state.candidates[x].elo_rating
        )
        stage2_survivors = sorted_survivors[
            : max(2, int(len(stage1_survivors) * self.stage2_keep_ratio))
        ]

        # Stage 3: dense pairwise
        stage3_pairs = [
            (stage2_survivors[i], stage2_survivors[j])
            for i in range(len(stage2_survivors))
            for j in range(i + 1, len(stage2_survivors))
            if not self.state.has_compared(stage2_survivors[i], stage2_survivors[j])
        ]
        if stage3_pairs:
            for result in self._execute_comparisons_parallel(stage3_pairs):
                self._update_from_comparison(result)
            vlm_calls += len(stage3_pairs)

        # Cross-batch: this batch's winner vs the running leaderboard
        cross_pairs = []
        if self.enable_cross_batch and self.state.batch_count > 0:
            batch_winner = max(
                stage2_survivors, key=lambda x: self.state.candidates[x].elo_rating
            )
            cross_pairs = [
                (batch_winner, prev_id)
                for prev_id in self.get_top_k(self.cross_batch_top_k)
                if prev_id not in all_ids
                and not self.state.has_compared(batch_winner, prev_id)
            ]
            if cross_pairs:
                for result in self._execute_comparisons_parallel(cross_pairs):
                    self._update_from_comparison(result)
                vlm_calls += len(cross_pairs)

        elapsed = time.time() - t0
        self.state.batch_count += 1
        self.state.total_vlm_calls += vlm_calls
        self.state.total_eval_time += elapsed
        self.state.update_best()

        return {
            "batch_id": batch_id,
            "num_images": len(image_paths),
            "stage1_survivors": len(stage1_survivors),
            "stage2_survivors": len(stage2_survivors),
            "vlm_calls": vlm_calls,
            "cross_batch_pairs": len(cross_pairs),
            "elapsed_time": elapsed,
            "rankings": self.get_rankings(),
            "current_best": self.get_best(),
            "current_best_seed": self.get_best_seed(),
        }
