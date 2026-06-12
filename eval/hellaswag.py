"""
eval/hellaswag.py — HellaSwag zero-shot (secundário, Fase C).

Likelihood scoring: para cada exemplo, escolhe a continuação com menor loss
média (por token do ending) dado o contexto. Loader mínimo próprio.

Uso:
    python -m eval.hellaswag --checkpoint checkpoints/hybrid_3_1/last.pt
"""

import argparse
import json
import os
import sys

import torch


@torch.no_grad()
def eval_hellaswag(model, model_cfg, device, max_examples: int = None) -> dict:
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("Rowan/hellaswag", split="validation")
    model.eval()

    correct = total = 0
    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        ctx_tokens = enc.encode_ordinary(ex["ctx"])
        label = int(ex["label"])

        best_idx, best_score = -1, float("inf")
        for idx, ending in enumerate(ex["endings"]):
            end_tokens = enc.encode_ordinary(" " + ending)
            full = (ctx_tokens + end_tokens)[-(model_cfg.max_seq_len + 1):]
            if len(full) < 2 or len(end_tokens) == 0:
                continue
            ids = torch.tensor([full[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([full[1:]], dtype=torch.long, device=device)
            # Avalia loss SÓ nas posições do ending (mascara o contexto com -1).
            ctx_len = max(0, len(full) - 1 - len(end_tokens))
            targets[:, :ctx_len] = -1
            _, loss = model(ids, targets)
            if loss is not None and loss.item() < best_score:
                best_score, best_idx = loss.item(), idx

        correct += int(best_idx == label)
        total += 1

    acc = correct / total if total else 0.0
    return {"hellaswag_accuracy": acc, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import HybridModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HybridModel(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(model.describe())
    print(json.dumps(eval_hellaswag(model, ckpt["model_cfg"], device, args.max_examples), indent=2))


if __name__ == "__main__":
    main()
