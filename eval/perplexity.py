"""
eval/perplexity.py — Perplexidade no split de validação FIXO do SlimPajama.

Usa o held-out determinístico do dataloader (data/dataloader.py), garantindo que
a perplexidade seja comparável entre variantes e entre runs.

Uso:
    python -m eval.perplexity --checkpoint checkpoints/hybrid_3_1/last.pt
"""

import argparse
import json
import math
import os
import sys

import torch


@torch.no_grad()
def eval_perplexity(model, model_cfg, train_cfg, device, n_batches: int = 50) -> dict:
    from data.dataloader import get_val_batches

    model.eval()

    amp = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    total_loss, total_tok = 0.0, 0
    for x, y in get_val_batches(train_cfg, model_cfg)[:n_batches]:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=amp, enabled=(device == "cuda")):
            _, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tok += x.numel()

    if total_tok == 0:
        return {"perplexity": float("nan"), "avg_loss": float("nan"), "tokens": 0}
    avg = total_loss / total_tok
    return {"perplexity": math.exp(avg), "avg_loss": avg, "tokens": total_tok}


def main():
    parser = argparse.ArgumentParser(description="Perplexidade no val fixo do SlimPajama")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_batches", type=int, default=50)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TrainConfig
    from model import HybridModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt["model_cfg"]
    train_cfg = ckpt.get("train_cfg", TrainConfig())
    train_cfg.batch_size = 8  # batch menor na avaliação
    model = HybridModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(model.describe())

    res = eval_perplexity(model, model_cfg, train_cfg, device, n_batches=args.n_batches)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
