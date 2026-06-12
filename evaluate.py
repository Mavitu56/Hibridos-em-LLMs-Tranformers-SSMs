"""
evaluate.py — Dispatcher de avaliação (fino) sobre o pacote eval/.

A lógica de cada benchmark vive em eval/{perplexity,lambada,hellaswag,mqar,ruler}.py.
Este script só carrega um checkpoint, roda os benchmarks pedidos e salva o JSON
com o nome da variante e o step.

Uso:
    python evaluate.py --checkpoint checkpoints/hybrid_3_1/last.pt \
        --benchmarks perplexity,mqar,lambada
"""

import argparse
import json
import os
import time

import torch

from config import TrainConfig
from model import HybridModel

from eval.perplexity import eval_perplexity
from eval.lambada import eval_lambada
from eval.hellaswag import eval_hellaswag
from eval.mqar import evaluate_mqar
from eval.ruler import evaluate_ruler


def main():
    parser = argparse.ArgumentParser(description="Avaliação de checkpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmarks", default="perplexity",
                        help="Lista separada por vírgula: perplexity,mqar,lambada,hellaswag,ruler")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--n_batches", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt["model_cfg"]
    train_cfg = ckpt.get("train_cfg", TrainConfig())
    train_cfg.batch_size = 8
    step = ckpt.get("step", 0)
    model = HybridModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(model.describe())

    pattern = model_cfg.block_pattern
    results = {
        "checkpoint": args.checkpoint,
        "step": step,
        "variant": f"{pattern.count('M')}:{pattern.count('A')}",
    }

    for name in [b.strip() for b in args.benchmarks.split(",")]:
        print(f"\nAvaliando {name}...")
        t0 = time.time()
        if name == "perplexity":
            out = eval_perplexity(model, model_cfg, train_cfg, device, n_batches=args.n_batches)
        elif name == "mqar":
            out = evaluate_mqar(model, device=device)
        elif name == "lambada":
            out = eval_lambada(model, model_cfg, device)
        elif name == "hellaswag":
            out = eval_hellaswag(model, model_cfg, device)
        elif name == "ruler":
            out = evaluate_ruler(model, task="niah", device=device, seq_len=1024, n_examples=100)
        else:
            print(f"  benchmark desconhecido: {name}")
            continue
        print(f"  {name}: {out}  ({time.time()-t0:.1f}s)")
        results[name] = out

    os.makedirs(args.out_dir, exist_ok=True)
    safe = results["variant"].replace(":", "-")
    out_path = os.path.join(args.out_dir, f"{safe}_step{step:07d}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
