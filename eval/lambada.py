"""
eval/lambada.py — LAMBADA zero-shot (secundário, Fase C).

Loader mínimo próprio (sem lm-eval-harness): dado o contexto, o modelo deve
prever a última palavra. Métrica = accuracy do último token (greedy).

Nota: o LAMBADA "oficial" mede a última PALAVRA (pode ser multi-token no BPE).
Aqui usamos a aproximação de último-token (comum em ablações rápidas estilo
nanoGPT). Para o número de paper final, prefira o lm-eval-harness.

Uso:
    python -m eval.lambada --checkpoint checkpoints/hybrid_3_1/last.pt
"""

import argparse
import json
import os
import sys

import torch


@torch.no_grad()
def eval_lambada(model, model_cfg, device, max_examples: int = None) -> dict:
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    model.eval()

    correct = total = 0
    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        tokens = enc.encode_ordinary(ex["text"])
        if len(tokens) < 2:
            continue
        tokens = tokens[-model_cfg.max_seq_len:]
        ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        logits, _ = model(ids)
        pred = logits[0, -1].argmax().item()
        correct += int(pred == tokens[-1])
        total += 1

    acc = correct / total if total else 0.0
    return {"lambada_accuracy": acc, "correct": correct, "total": total}


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
    print(json.dumps(eval_lambada(model, ckpt["model_cfg"], device, args.max_examples), indent=2))


if __name__ == "__main__":
    main()
