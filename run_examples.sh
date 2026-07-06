PROMPT="a photo of four giraffes"

## Run Command (# method: bon/flash-bon, model: wan-1.3b/wan-14b/flux-dev)
CUDA_VISIBLE_DEVICES=0 python run.py --method flash-bon --prompt "$PROMPT" --budget_seconds 150 --model_key flux-dev