"""
train.py — Loop de treino estilo nanoGPT para os modelos híbridos SSM-Transformer.

Pontos da spec (Tarefas 3 e 6, restrições da §8):
  - IMPORTÁVEL e chamável por função: train(variant_name, cfg) -> metrics.
    NÃO depende de subprocesso (!python não herda o backend/patches no Colab).
    Há uma CLI fina por cima da função.
  - bf16 (autocast) por padrão; em GPU sem bf16 (T4) cai para fp32 e AVISA.
    Nunca fp16 puro em Mamba.
  - Gradient clipping (norma global) e init escalado por profundidade (este
    último vive em model.py, aplicado na construção do modelo).
  - AdamW (0.9, 0.95), weight decay 0.1 sem decay em normas/bias/params 1-D.
  - LR cosseno com warmup; mesmo orçamento de tokens entre variantes (max_steps
    derivado de max_tokens / tokens_por_step).
  - Checkpoint a cada N steps + RESUME automático do último checkpoint (assume
    que a sessão do Colab vai cair). Aponte out_dir para o Google Drive.
  - Logging por intervalo: train loss, val loss/perplexidade, tokens/s, pico de
    memória, tempo/step.

Uso (Python):
    from config import TrainConfig
    from train import train
    metrics = train("hybrid_3_1", TrainConfig(out_dir="/content/drive/MyDrive/ckpts"))

Uso (CLI):
    python train.py --variant hybrid_3_1 --out_dir checkpoints/hybrid_3_1 --max_steps 5000
"""

import argparse
import math
import os
import time

import torch

from config import ModelConfig, TrainConfig, resolve_variant
from data.dataloader import make_dataloader, get_val_batches
from model import HybridModel


# ---------------------------------------------------------------------------
# Precisão
# ---------------------------------------------------------------------------

def resolve_dtype(train_cfg: TrainConfig, device: str) -> torch.dtype:
    """
    bf16 se a GPU suportar; senão fp32 com aviso. Nunca fp16 puro (instável em Mamba).
    """
    if device != "cuda":
        print("[precisão] CPU detectada — usando fp32.")
        return torch.float32
    if train_cfg.dtype == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    print(
        "[precisão] bf16 indisponível nesta GPU (ex.: T4) — caindo para fp32. "
        "Nunca usamos fp16 puro em Mamba (instabilidade numérica)."
    )
    return torch.float32


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_lr(step: int, max_steps: int, train_cfg: TrainConfig) -> float:
    """Cosine decay com warmup linear (frações definidas em TrainConfig)."""
    warmup = max(1, int(train_cfg.warmup_frac * max_steps))
    min_lr = train_cfg.learning_rate * train_cfg.min_lr_frac
    if step < warmup:
        return train_cfg.learning_rate * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, (max_steps - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (train_cfg.learning_rate - min_lr)


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, tokens_seen, train_cfg, model_cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_last.tmp.pt")
    path = os.path.join(out_dir, "last.pt")
    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "model_cfg": model_cfg,
            "train_cfg": train_cfg,
            # Backend grava nomes de parâmetros DIFERENTES no mixer Mamba
            # (kernels: mixer.mamba.* ; torch: mixer.mixer.*). Registramos para
            # o resume recusar um checkpoint de outro backend em vez de quebrar.
            "mamba_backend": os.environ.get("MAMBA_BACKEND", "torch"),
        },
        tmp,
    )
    # Escrita atômica: evita checkpoint corrompido se a sessão cair no meio do save.
    os.replace(tmp, path)
    print(f"  checkpoint -> {path} (step {step})")


def load_checkpoint(out_dir, model, optimizer, device):
    """Retoma do último checkpoint, se existir. Devolve (step, tokens_seen)."""
    path = os.path.join(out_dir, "last.pt")
    if not os.path.exists(path):
        return 0, 0
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # O nome dos parâmetros do mixer Mamba depende do backend (kernels:
    # mixer.mamba.* ; torch: mixer.mixer.*). Um checkpoint de outro backend
    # NÃO carrega (load_state_dict quebra). Detectamos e começamos do zero em
    # vez de derrubar a run — o checkpoint antigo é sobrescrito no 1º save.
    ckpt_backend = ckpt.get("mamba_backend")
    cur_backend = os.environ.get("MAMBA_BACKEND", "torch")
    # Só há conflito se a variante tem blocos Mamba (o GQA é igual nos dois
    # backends — attn_only treinado em 'torch' carrega em 'kernels' sem dor).
    has_mamba = "M" in getattr(ckpt.get("model_cfg"), "block_pattern", [])
    if has_mamba and ckpt_backend is not None and ckpt_backend != cur_backend:
        print(
            f"[resume] checkpoint em {path} foi salvo com backend "
            f"'{ckpt_backend}', mas o ativo é '{cur_backend}'. "
            f"Pesos incompatíveis — IGNORANDO o checkpoint e começando do zero."
        )
        return 0, 0
    try:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
    except RuntimeError as e:
        # Rede de segurança p/ checkpoints antigos sem o campo mamba_backend.
        print(
            f"[resume] FALHA ao carregar {path} ({type(e).__name__}). "
            f"Provável incompatibilidade de backend/arquitetura. "
            f"IGNORANDO o checkpoint e começando do zero."
        )
        return 0, 0
    step = ckpt.get("step", 0)
    tokens_seen = ckpt.get("tokens_seen", 0)
    print(f"[resume] retomando de {path}: step {step}, {tokens_seen/1e9:.3f}B tokens")
    return step, tokens_seen


# ---------------------------------------------------------------------------
# Avaliação de validação (perplexidade no held-out fixo)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_val_loss(model, model_cfg, train_cfg, device, amp_dtype, n_batches):
    model.eval()
    # Val cacheado em memória: evita reabrir o stream HF a cada avaliação
    # (lento; uma falha de rede aqui derrubaria a run de treino inteira).
    batches = get_val_batches(train_cfg, model_cfg)[:n_batches]
    total_loss, total_tok = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tok += x.numel()
    model.train()
    if total_tok == 0:
        return float("nan"), float("nan")
    avg = total_loss / total_tok
    return avg, math.exp(min(avg, 20))  # clamp p/ não estourar exp em loss inicial alta


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def train(variant_name: str, cfg: TrainConfig = None) -> dict:
    """
    Treina uma variante. Importável e chamável inline (sem subprocesso).

    Args:
        variant_name: nome da spec ("hybrid_3_1") ou razão ("3:1").
        cfg: TrainConfig; se None, usa os defaults. out_dir deve apontar p/ o
             Drive no Colab para sobreviver a quedas de sessão.

    Returns:
        dict de métricas finais (step, val_loss, val_ppl, tokens_seen, ...).
    """
    cfg = cfg or TrainConfig()
    pattern = resolve_variant(variant_name)
    model_cfg = ModelConfig(block_pattern=pattern)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = resolve_dtype(cfg, device)
    out_dir = cfg.out_dir
    print(f"[device] {device}  [dtype] {amp_dtype}")
    print(f"[backend] MAMBA_BACKEND={os.environ.get('MAMBA_BACKEND', 'torch (default)')}")
    if device == "cuda":
        torch.cuda.empty_cache()  # libera resíduos de runs anteriores na sessão

    # --- Micro-batch adaptativo p/ backend torch puro com blocos Mamba ---
    # O torch_forward do Mamba2Mixer materializa um tensor O(B·T·chunk·h·p) em
    # fp32 (B=16, T=1024 → alocação única de ~16 GiB; OOM observado em A100
    # 40GB na Fase A). Reduzimos batch_size e aumentamos grad_accum na MESMA
    # proporção: tokens/step e batch efetivo idênticos => comparável entre
    # variantes e backends. Ver TrainConfig.mamba_torch_microbatch.
    backend = os.environ.get("MAMBA_BACKEND", "torch")
    target_mb = cfg.mamba_torch_microbatch
    if backend == "torch" and "M" in pattern and target_mb > 0:
        while cfg.batch_size > target_mb and cfg.batch_size % 2 == 0:
            cfg.batch_size //= 2
            cfg.grad_accumulation_steps *= 2
        print(f"[micro-batch] backend torch + blocos Mamba: batch_size={cfg.batch_size}, "
              f"grad_accum={cfg.grad_accumulation_steps} (tokens/step inalterado)")

    # --- Modelo (init escalado por profundidade já aplicado no construtor) ---
    model = HybridModel(model_cfg).to(device)
    print(model.describe())
    print(f"Parâmetros totais: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")

    # --- Otimizador: sem weight decay em params 1-D (normas/bias) ---
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )

    # --- Orçamento de tokens -> max_steps (mesmo orçamento entre variantes) ---
    tokens_per_step = cfg.batch_size * cfg.grad_accumulation_steps * cfg.block_size
    if cfg.max_steps and cfg.max_steps > 0:
        max_steps = cfg.max_steps
    else:
        max_steps = max(1, cfg.max_tokens // tokens_per_step)
    print(f"[orçamento] {tokens_per_step} tok/step × {max_steps} steps "
          f"= {tokens_per_step*max_steps/1e9:.2f}B tokens\n")

    # --- Resume automático ---
    step, tokens_seen = load_checkpoint(out_dir, model, optimizer, device)

    # --- Dados ---
    train_loader = make_dataloader(cfg, model_cfg, split="train")
    data_iter = iter(train_loader)

    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    last_val = {}

    print("Iniciando treino...\n")
    while step < max_steps:
        lr = get_lr(step, max_steps, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(cfg.grad_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accumulation_steps
            loss.backward()
            step_loss += loss.item()

        # Gradient clipping (norma global = grad_clip)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        tokens_seen += tokens_per_step
        step += 1

        # Guarda contra 'nan' — para cedo com diagnóstico em vez de queimar GPU.
        if not math.isfinite(step_loss):
            raise FloatingPointError(
                f"loss não-finita (nan/inf) no step {step} da variante "
                f"'{variant_name}'. Verifique init escalado, grad clip e dtype "
                f"(bf16/fp32, nunca fp16 puro em Mamba)."
            )

        # --- Logging de treino ---
        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            tok_s = cfg.log_interval * tokens_per_step / dt if dt > 0 else 0.0
            peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
            print(
                f"step {step:6d}/{max_steps} | loss {step_loss:.4f} | lr {lr:.2e} | "
                f"{tok_s/1e3:.1f}k tok/s | {dt/cfg.log_interval*1e3:.0f} ms/step | "
                f"peak {peak_gb:.1f}GB | {tokens_seen/1e9:.2f}B tok"
            )
            t0 = time.time()

        # --- Avaliação de validação ---
        if step % cfg.eval_interval == 0:
            vloss, vppl = eval_val_loss(model, model_cfg, cfg, device, amp_dtype, cfg.eval_batches)
            last_val = {"val_loss": vloss, "val_ppl": vppl}
            print(f"  [val] step {step} | val_loss {vloss:.4f} | perplexidade {vppl:.2f}")
            t0 = time.time()  # não contar o tempo de eval no tok/s

        # --- Checkpoint (resume no Drive) ---
        if step % cfg.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, tokens_seen, cfg, model_cfg, out_dir)

    # Checkpoint e avaliação finais
    save_checkpoint(model, optimizer, step, tokens_seen, cfg, model_cfg, out_dir)
    vloss, vppl = eval_val_loss(model, model_cfg, cfg, device, amp_dtype, cfg.eval_batches)
    print(f"\nTreino concluído. val_loss {vloss:.4f} | perplexidade {vppl:.2f}")

    return {
        "variant": variant_name,
        "step": step,
        "tokens_seen": tokens_seen,
        "val_loss": vloss,
        "val_ppl": vppl,
        **last_val,
    }


# ---------------------------------------------------------------------------
# CLI fina
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Treino de modelos híbridos SSM-Transformer")
    parser.add_argument("--variant", type=str, default="hybrid_3_1",
                        help="Nome da variante (hybrid_3_1) ou razão (3:1)")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="0 => derivar de max_tokens")
    parser.add_argument("--max_tokens", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig(out_dir=args.out_dir, max_steps=args.max_steps)
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    train(args.variant, cfg)


if __name__ == "__main__":
    main()
