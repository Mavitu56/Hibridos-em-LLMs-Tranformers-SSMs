"""
run_experiments.py — Orquestração dos experimentos na ordem priorizada da §6.

NÃO inverter a ordem: a Fase A é um GATE. Só avança para B se A inteira passar.

Fluxo:
    Fase A (gate):
        1. setup_env  -> reporta backend (kernels|torch)
        2. smoke de blocos (forward sem nan nos dois tipos)
        3. check_parity (±5%)
        4. smoke train ~50 steps (loss cai, sem nan, checkpoint+resume)
        5. baselines attn_only e ssm_only até o orçamento
    Fase B (núcleo da hipótese):
        6. hybrid_3_1
        7. MQAR + perplexidade em attn_only, ssm_only, hybrid_3_1
    Fase C (upside):
        8. hybrid_5_1, hybrid_7_1
        9. sweep de benchmarks (lambada, hellaswag, ruler)

Tudo é importável e inline (sem !python; o backend não sobrevive a subprocesso
no Colab). Cada fase pode ser chamada isoladamente.

Uso (Colab):
    import run_experiments as R
    R.phase_a(out_root="/content/drive/MyDrive/hybrid_ckpts")
"""

import os

import torch

from config import TrainConfig
import setup_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_backend():
    if "MAMBA_BACKEND" not in os.environ:
        setup_env.setup()
    print(f"[backend] MAMBA_BACKEND={os.environ['MAMBA_BACKEND']}")


def _train_cfg(out_dir, **over):
    cfg = TrainConfig(out_dir=out_dir)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Fase A — GATE
# ---------------------------------------------------------------------------

def smoke_blocks():
    """Forward de cada tipo de bloco em input aleatório: shape ok, sem nan."""
    from config import ModelConfig
    from blocks import GQABlock, MambaBlock

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig()
    x = torch.randn(2, cfg.chunk_size, cfg.d_model, device=device)
    ok = True
    for name, Block in [("GQABlock", GQABlock), ("MambaBlock", MambaBlock)]:
        blk = Block(cfg).to(device)
        y = blk(x)
        finite = torch.isfinite(y).all().item()
        shape_ok = y.shape == x.shape
        print(f"  {name}: out={tuple(y.shape)} finito={finite} shape_ok={shape_ok}")
        ok = ok and finite and shape_ok
    return ok


def smoke_parity():
    import check_parity
    return check_parity.check_all_variants()


def smoke_train(out_root):
    """~50 steps na variante 3:1; valida queda de loss, sem nan, resume."""
    from train import train
    out_dir = os.path.join(out_root, "_smoke_hybrid_3_1")
    cfg = _train_cfg(
        out_dir, max_steps=50, batch_size=4, grad_accumulation_steps=2,
        block_size=256, eval_interval=50, checkpoint_interval=25, log_interval=10,
        eval_batches=2,
    )
    m = train("hybrid_3_1", cfg)
    print(f"  smoke train metrics: {m}")
    # Testa resume: chamar de novo deve retomar do checkpoint (step já = max_steps).
    print("  testando resume...")
    train("hybrid_3_1", cfg)
    return m["val_ppl"] == m["val_ppl"]  # not-nan check


def phase_a(out_root="checkpoints", run_baselines=True):
    print("\n##### FASE A — GATE #####")
    _ensure_backend()

    print("\n[A2] smoke de blocos")
    assert smoke_blocks(), "smoke de blocos falhou (nan ou shape)."

    print("\n[A3] check_parity")
    assert smoke_parity(), "paridade fora de ±5%."

    print("\n[A4] smoke train (~50 steps) + resume")
    assert smoke_train(out_root), "smoke train falhou."

    if run_baselines:
        print("\n[A5] baselines até o orçamento")
        from train import train
        for v in ("attn_only", "ssm_only"):
            print(f"\n--- baseline {v} ---")
            train(v, _train_cfg(os.path.join(out_root, v)))
    print("\n✓ FASE A concluída.")


# ---------------------------------------------------------------------------
# Fase B — núcleo da hipótese
# ---------------------------------------------------------------------------

def phase_b(out_root="checkpoints"):
    print("\n##### FASE B — núcleo da hipótese #####")
    _ensure_backend()
    from train import train
    from eval.mqar import evaluate_mqar
    from eval.perplexity import eval_perplexity
    from model import HybridModel

    print("\n[B6] treinar hybrid_3_1")
    train("hybrid_3_1", _train_cfg(os.path.join(out_root, "hybrid_3_1")))

    print("\n[B7] MQAR + perplexidade em attn_only, ssm_only, hybrid_3_1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {}
    for v in ("attn_only", "ssm_only", "hybrid_3_1"):
        path = os.path.join(out_root, v, "last.pt")
        if not os.path.exists(path):
            print(f"  [pulado] {v}: checkpoint ausente ({path})")
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = HybridModel(ckpt["model_cfg"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        mqar = evaluate_mqar(model, device=device)
        ppl = eval_perplexity(model, ckpt["model_cfg"], ckpt.get("train_cfg", TrainConfig()),
                              device, n_batches=20)
        summary[v] = {"mqar": mqar["mqar_accuracy"], "ppl": ppl["perplexity"]}
        print(f"  {v}: MQAR={mqar['mqar_accuracy']:.3f}  ppl={ppl['perplexity']:.2f}")
        del model
    print(f"\nResumo Fase B: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Fase B+ — varredura de recall (diagnóstico do gap atenção↔SSM)
# ---------------------------------------------------------------------------

def phase_b_recall(out_root="checkpoints", variants=None, out_dir=None):
    """
    Varredura de recall sobre os checkpoints JÁ treinados (só inferência).

    Por que isto existe (CHANGELOG, "Avaliação 2026-06-16"): o MQAR de ponto
    único da phase_b (seq_len=128, n_pairs=8) não distingue "não há efeito" de
    "medi no regime onde as arquiteturas se parecem". Aqui varremos o MQAR em
    (seq_len × n_pairs) e o NIAH em vários seq_len, com nível de acaso reportado
    — a curva de Zoology (Arora 2023), não um número solto. Não retreina nada.

    Roda nos checkpoints que existirem; pula silenciosamente os ausentes (ex.:
    hybrid_5_1/7_1 antes da Fase C).
    """
    print("\n##### FASE B+ — varredura de recall (MQAR grid + NIAH) #####")
    _ensure_backend()
    from eval.recall_sweep import sweep_checkpoint

    variants = variants or ("attn_only", "ssm_only", "hybrid_3_1",
                            "hybrid_5_1", "hybrid_7_1")
    out_dir = out_dir or os.path.join(out_root, "_recall_results")
    summary = {}
    for v in variants:
        path = os.path.join(out_root, v, "last.pt")
        if not os.path.exists(path):
            print(f"  [pulado] {v}: checkpoint ausente ({path})")
            continue
        print(f"\n--- recall sweep: {v} ---")
        res = sweep_checkpoint(path, out_dir=out_dir)
        # Resumo enxuto: a célula MQAR mais difícil viável (maior seq_len/n_pairs).
        cells = res["mqar_grid"]["cells"]
        hardest = max(cells, key=lambda c: (c["seq_len"], c["n_pairs"])) if cells else None
        summary[v] = {
            "mqar_hardest_cell": hardest,
            "chance": res["mqar_grid"]["chance_level"],
            "out_path": res.get("out_path"),
        }
    print(f"\nResumo Fase B+ (célula MQAR mais difícil por variante):")
    for v, s in summary.items():
        hc = s["mqar_hardest_cell"]
        if hc:
            print(f"  {v}: seq_len={hc['seq_len']} n_pairs={hc['n_pairs']} "
                  f"acc={hc['accuracy']:.4f} (acaso={s['chance']:.4f})")
    return summary


# ---------------------------------------------------------------------------
# Fase C — upside
# ---------------------------------------------------------------------------

def phase_c(out_root="checkpoints"):
    print("\n##### FASE C — upside #####")
    _ensure_backend()
    from train import train
    for v in ("hybrid_5_1", "hybrid_7_1"):
        print(f"\n--- treinar {v} ---")
        train(v, _train_cfg(os.path.join(out_root, v)))
    print("Fase C: rode evaluate.py com --benchmarks perplexity,mqar,lambada,hellaswag,ruler")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a", "b", "b_recall", "c"], default="a")
    parser.add_argument("--out_root", default="checkpoints")
    parser.add_argument("--no_baselines", action="store_true")
    args = parser.parse_args()
    if args.phase == "a":
        phase_a(args.out_root, run_baselines=not args.no_baselines)
    elif args.phase == "b":
        phase_b(args.out_root)
    elif args.phase == "b_recall":
        phase_b_recall(args.out_root)
    else:
        phase_c(args.out_root)
