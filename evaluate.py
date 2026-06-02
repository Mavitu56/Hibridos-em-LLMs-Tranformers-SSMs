"""
evaluate.py — Avaliação de checkpoints em benchmarks de linguagem

Benchmarks suportados:
    perplexity  — Perplexidade no validation split do SlimPajama
    lambada     — Zero-shot accuracy no LAMBADA
    hellaswag   — Zero-shot accuracy no HellaSwag

Uso:
    python evaluate.py --checkpoint checkpoints/3:1/step_0100000.pt
    python evaluate.py --checkpoint ckpt.pt --benchmarks perplexity,lambada
"""

import argparse
import json
import math
import os
import sys
import time

import torch

from config import ModelConfig, TrainConfig
from model import HybridModel


# ---------------------------------------------------------------------------
# Perplexidade
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity(
    model: HybridModel,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: str,
    n_batches: int = 50,
) -> dict:
    """
    Estima perplexidade no validation split do SlimPajama.
    Usa n_batches × batch_size sequências para manter custo controlado.
    """
    from data.dataloader import make_dataloader

    loader   = make_dataloader(train_cfg, model_cfg, split="validation", num_workers=0)
    data_it  = iter(loader)
    model.eval()

    total_loss  = 0.0
    total_tokens = 0

    for _ in range(n_batches):
        try:
            x, y = next(data_it)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(x, y)
        total_loss   += loss.item() * x.numel()
        total_tokens += x.numel()

    avg_loss    = total_loss / total_tokens
    perplexity  = math.exp(avg_loss)
    return {"perplexity": perplexity, "avg_loss": avg_loss, "tokens": total_tokens}


# ---------------------------------------------------------------------------
# LAMBADA zero-shot
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_lambada(model: HybridModel, model_cfg: ModelConfig, device: str) -> dict:
    """
    Zero-shot accuracy no LAMBADA: o modelo deve atribuir maior probabilidade
    ao último token correto da passagem.

    Formato: dado o contexto, prever a última palavra.
    """
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds  = load_dataset("EleutherAI/lambada_openai", split="test")
    model.eval()

    correct = 0
    total   = 0

    for example in ds:
        text   = example["text"]
        tokens = enc.encode_ordinary(text)
        if len(tokens) < 2:
            continue

        # Trunca ao max_seq_len
        tokens = tokens[-model_cfg.max_seq_len:]
        ids    = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)

        logits, _ = model(ids)
        # Probabilidade do último token real
        last_logits   = logits[0, -1]  # (vocab_size,)
        predicted_tok = last_logits.argmax().item()

        if predicted_tok == tokens[-1]:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"lambada_accuracy": accuracy, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# HellaSwag zero-shot
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_hellaswag(model: HybridModel, model_cfg: ModelConfig, device: str) -> dict:
    """
    Zero-shot accuracy no HellaSwag: escolhe a continuação com menor
    log-perplexidade dado o contexto (likelihood scoring).
    """
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds  = load_dataset("Rowan/hellaswag", split="validation")
    model.eval()

    correct = 0
    total   = 0

    for example in ds:
        ctx_tokens = enc.encode_ordinary(example["ctx"])
        endings    = example["endings"]
        label      = int(example["label"])

        best_idx  = -1
        best_score = float("inf")  # menor loss = melhor candidato

        for idx, ending in enumerate(endings):
            end_tokens = enc.encode_ordinary(" " + ending)  # espaço inicial padrão
            full       = ctx_tokens + end_tokens
            full       = full[-model_cfg.max_seq_len - 1:]  # trunca preservando fim

            ids     = torch.tensor([full[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([full[1:]],  dtype=torch.long, device=device)

            # Mascara o contexto: só avalia loss nas posições do ending
            ctx_len = max(0, len(full) - 1 - len(end_tokens))
            # Substitui labels do contexto por -1 (ignorado no cross_entropy)
            targets[:, :ctx_len] = -1

            _, loss = model(ids, targets)
            if loss is not None and loss.item() < best_score:
                best_score = loss.item()
                best_idx   = idx

        if best_idx == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"hellaswag_accuracy": accuracy, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

BENCHMARK_FNS = {
    "perplexity": eval_perplexity,
    "lambada":    eval_lambada,
    "hellaswag":  eval_hellaswag,
}


def main():
    parser = argparse.ArgumentParser(description="Avaliação de checkpoints")
    parser.add_argument("--checkpoint",  required=True,              help="Caminho do checkpoint .pt")
    parser.add_argument("--benchmarks",  default="perplexity",       help="Lista separada por vírgula")
    parser.add_argument("--out_dir",     default="results",          help="Diretório para salvar JSON")
    parser.add_argument("--n_batches",   type=int, default=50,       help="Batches para perplexidade")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Carrega checkpoint
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg = ckpt["model_cfg"]
    step      = ckpt.get("step", 0)
    model     = HybridModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(model.describe())

    train_cfg = TrainConfig(batch_size=8)  # batch menor para avaliação
    requested = [b.strip() for b in args.benchmarks.split(",")]

    results = {"checkpoint": args.checkpoint, "step": step}
    pattern = model_cfg.block_pattern
    n_attn  = pattern.count('A')
    n_mamba = pattern.count('M')
    results["variant"] = f"{n_mamba}:{n_attn}"

    for name in requested:
        if name not in BENCHMARK_FNS:
            print(f"Benchmark desconhecido: '{name}'. Opções: {list(BENCHMARK_FNS.keys())}")
            continue
        print(f"\nAvaliando {name}...")
        t0 = time.time()
        fn = BENCHMARK_FNS[name]

        if name == "perplexity":
            out = fn(model, model_cfg, train_cfg, device, n_batches=args.n_batches)
        else:
            out = fn(model, model_cfg, device)

        elapsed = time.time() - t0
        print(f"  {name}: {out}  ({elapsed:.1f}s)")
        results[name] = out

    # Salva JSON com nome da variante e step
    os.makedirs(args.out_dir, exist_ok=True)
    variant_safe = results["variant"].replace(":", "-")
    out_path     = os.path.join(args.out_dir, f"{variant_safe}_step{step:07d}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
