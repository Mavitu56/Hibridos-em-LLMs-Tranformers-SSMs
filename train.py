"""
train.py — Loop de treino estilo nanoGPT para modelos híbridos SSM-Transformer

Uso:
    python train.py --variant 3:1 --out_dir checkpoints/3:1 --max_steps 100000
"""

import argparse
import math
import os
import time

import torch

from config import ModelConfig, TrainConfig, VARIANTS
from data.dataloader import make_dataloader
from model import HybridModel


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine decay com warmup linear — padrão da literatura de LMs."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    if step > cfg.lr_decay_steps:
        return cfg.learning_rate * 0.1  # piso em 10% da LR máxima
    progress = (step - cfg.warmup_steps) / (cfg.lr_decay_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.learning_rate * 0.1 + coeff * cfg.learning_rate * 0.9


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model: HybridModel, optimizer, step: int, loss: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step_{step:07d}.pt")
    torch.save(
        {
            "step":        step,
            "loss":        loss,
            "model_state": model.state_dict(),
            "model_cfg":   model.cfg,
            "optim_state": optimizer.state_dict(),
        },
        path,
    )
    print(f"  checkpoint → {path}")


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def train(args):
    # --- Config ---
    if args.variant not in VARIANTS:
        raise ValueError(
            f"Variante '{args.variant}' desconhecida. "
            f"Opções: {list(VARIANTS.keys())}"
        )

    model_cfg = ModelConfig(block_pattern=VARIANTS[args.variant])
    train_cfg = TrainConfig(out_dir=args.out_dir)
    if args.max_steps:
        train_cfg.lr_decay_steps = args.max_steps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Modelo ---
    model = HybridModel(model_cfg).to(device)
    print(model.describe())
    print(f"Parâmetros totais: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    # torch.compile acelera ~20-30% em GPUs Ampere+; fallback se não disponível
    try:
        model = torch.compile(model)
        print("torch.compile() ativado.")
    except Exception as e:
        print(f"torch.compile() indisponível ({e}), continuando sem compilação.")

    # --- Otimizador ---
    # Separa parâmetros com e sem weight decay (bias e norms ficam sem decay)
    decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )

    # --- Dados ---
    train_loader = make_dataloader(train_cfg, model_cfg, split="train")
    data_iter    = iter(train_loader)

    # --- Loop ---
    step          = 0
    tokens_so_far = 0
    t0            = time.time()
    accum_loss    = 0.0
    max_steps     = args.max_steps or int(1e18)  # sem limite se não especificado

    print("Iniciando treino...\n")
    model.train()

    while step < max_steps:
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Acumulação de gradiente
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(train_cfg.grad_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss    = loss / train_cfg.grad_accumulation_steps

            loss.backward()
            step_loss += loss.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        tokens_so_far += (
            train_cfg.batch_size
            * train_cfg.grad_accumulation_steps
            * model_cfg.max_seq_len
        )

        # --- Logging ---
        if step % train_cfg.log_interval == 0:
            t1       = time.time()
            dt       = t1 - t0
            tok_per_s = (
                train_cfg.batch_size
                * train_cfg.grad_accumulation_steps
                * model_cfg.max_seq_len
                * train_cfg.log_interval
                / dt
            )
            print(
                f"step {step:7d} | loss {step_loss:.4f} | "
                f"lr {lr:.2e} | {tok_per_s/1e3:.1f}k tok/s | "
                f"{tokens_so_far/1e9:.2f}B tokens"
            )
            t0 = t1

        # --- Checkpoint e avaliação ---
        if step > 0 and step % train_cfg.eval_interval == 0:
            save_checkpoint(model, optimizer, step, step_loss, train_cfg.out_dir)

        step += 1

    # Checkpoint final
    save_checkpoint(model, optimizer, step, step_loss, train_cfg.out_dir)
    print("Treino concluído.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino de modelos híbridos SSM-Transformer")
    parser.add_argument(
        "--variant",
        type=str,
        default="3:1",
        help=f"Variante arquitetural. Opções: {list(VARIANTS.keys())}",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints",
        help="Diretório para salvar checkpoints",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Número máximo de steps (sobrescreve lr_decay_steps)",
    )
    args = parser.parse_args()
    train(args)
