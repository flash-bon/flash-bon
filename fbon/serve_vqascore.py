"""VQAScore HTTP server (Flask) for the final-eval step.

Uses the pip-installed ``t2v_metrics`` (imported after CUDA_VISIBLE_DEVICES is set by the launcher).

    GET  /health -> {"status": "ok" | "loading" | "error"}
    POST /infer  {"image_path","prompt"} -> {"score": float, "time": float}
"""

import argparse
import threading
import time

from flask import Flask, jsonify, request

app = Flask(__name__)
_STATE = {"model": None, "status": "loading", "error": None}


def _load(model_name):
    try:
        import t2v_metrics  # heavy; imported after the GPU is pinned by the launcher
        scorer = t2v_metrics.VQAScore(model=model_name)
        # warm up / force weight load before reporting ready
        from PIL import Image
        tmp = "/tmp/vqascore_warmup.png"
        Image.new("RGB", (64, 64), (127, 127, 127)).save(tmp)
        _ = scorer(images=[tmp], texts=["warmup"])
        _STATE["model"] = scorer
        _STATE["status"] = "ok"
        print("[vqascore] model ready:", model_name, flush=True)
    except Exception as e:  # report via /health rather than crashing silently
        _STATE["status"] = "error"
        _STATE["error"] = repr(e)
        print("[vqascore] load error:", repr(e), flush=True)


@app.route("/health", methods=["GET"], strict_slashes=False)
def health():
    if _STATE["status"] == "ok":
        return jsonify({"status": "ok"}), 200
    if _STATE["status"] == "error":
        return jsonify({"status": "error", "error": _STATE["error"]}), 500
    return jsonify({"status": "loading"}), 503


@app.route("/infer", methods=["POST"], strict_slashes=False)
def infer():
    if _STATE["status"] != "ok":
        return jsonify({"error": f"model {_STATE['status']}"}), 503
    data = request.get_json(force=True)
    image_path, prompt = data.get("image_path"), data.get("prompt")
    if not image_path or not prompt:
        return jsonify({"error": "image_path and prompt required"}), 400
    t0 = time.time()
    score = _STATE["model"](images=[image_path], texts=[prompt])
    if hasattr(score, "item"):
        score = score.item()
    return jsonify({"score": float(score), "time": time.time() - t0}), 200


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--vqascore_model", default="qwen2.5-vl-7b")
    args = p.parse_args()
    threading.Thread(target=_load, args=(args.vqascore_model,), daemon=True).start()
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
