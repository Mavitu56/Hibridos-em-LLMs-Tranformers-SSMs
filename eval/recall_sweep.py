"""
eval/recall_sweep.py — Varredura de recall associativo (MQAR) e long-context (RULER).

MOTIVAÇÃO (registrada no CHANGELOG, seção "Avaliação 2026-06-16").
A hipótese central do TCC — "SSM puro colapsa no recall associativo de longo
alcance; um pouco de atenção recupera" (Arora et al. 2023, "Zoology") — NÃO se
manifesta num único ponto de medição. Em Zoology o gap atenção↔SSM *abre* à
medida que o comprimento da sequência e o número de pares (kv) crescem; medir só
em (seq_len=128, n_pairs=8), como faz evaluate.py com os defaults, é justamente o
regime em que as arquiteturas mais se parecem. Este módulo varre o MQAR num GRID
de (seq_len, n_pairs) e o NIAH do RULER em vários seq_len, reportando o nível de
ACASO (chance) e as contagens por célula — o que transforma um número solto numa
curva de trade-off interpretável.

Reaproveita os geradores/avaliadores já validados por selftest (eval/mqar.py,
eval/ruler.py) SEM alterar a interface pública deles. A semântica de avaliação é
idêntica: argmax sobre TODO o vocabulário do modelo (50257), comparado às
posições de resposta (accuracy@1). Só variam os parâmetros do gerador.

Uso (inline, Colab — preferido; o backend Mamba não sobrevive a subprocesso):
    from eval.recall_sweep import sweep_checkpoint
    res = sweep_checkpoint("/content/drive/MyDrive/hybrid_ckpts/ssm_only/last.pt")

Uso (CLI):
    python -m eval.recall_sweep --checkpoint .../last.pt --out_dir results
    python -m eval.recall_sweep --selftest
"""

import argparse
import json
import os
import sys
import time

import torch

# Imports relativos ao pacote eval/ — reusa os geradores já testados.
from eval.mqar import generate_mqar_examples
from eval.ruler import evaluate_ruler


# ---------------------------------------------------------------------------
# Grids padrão (ajustáveis por CLI/kwargs)
# ---------------------------------------------------------------------------
# seq_len limitado por ModelConfig.max_seq_len (2048); usamos até 1024 para
# manter custo de inferência baixo e ainda cobrir uma década de comprimentos.
# n_pairs cresce em potências de 2; o gerador exige 2*n_pairs + 1 (SEP) <= seq_len
# para caber o prefixo de pares — células inviáveis são puladas e marcadas.
DEFAULT_SEQ_LENS = (64, 128, 256, 512, 1024)
DEFAULT_N_PAIRS = (4, 8, 16, 32, 64)
DEFAULT_VOCAB_SIZE = 512        # mesmo default do evaluate.py (comparabilidade)
DEFAULT_N_EXAMPLES = 512        # por célula; suficiente p/ estabilizar accuracy@1
DEFAULT_NIAH_SEQ_LENS = (256, 512, 1024)


def _chance_level(vocab_size: int) -> float:
    """
    Nível de acaso da accuracy@1 no MQAR/NIAH. A resposta é um token de VALOR,
    sorteado uniformemente em [0, vocab_size); o modelo prediz sobre todo o
    vocabulário, então o acaso ingênuo é 1/vocab_size. Reportar isto é o que
    torna a acurácia absoluta interpretável (ressalva do CHANGELOG).
    """
    return 1.0 / vocab_size


@torch.no_grad()
def _eval_mqar_cell(model, seq_len, n_pairs, vocab_size, n_examples,
                    batch_size, device):
    """
    Avalia UMA célula do grid MQAR. Mesma semântica de eval.mqar.evaluate_mqar
    (model(x[:,:-1]) vs labels[:,:-1], argmax no vocab inteiro), mas separada
    aqui para variar seq_len/n_pairs e devolver contagens cruas.
    Retorna None se a célula for inviável (prefixo não cabe em seq_len).
    """
    # Prefixo = 2*n_pairs (pares) + 1 (SEP); precisa sobrar >=2 p/ uma query.
    if 2 * n_pairs + 1 + 2 > seq_len:
        return None

    model.eval()
    inputs, labels, total_vocab = generate_mqar_examples(
        n_examples=n_examples, seq_len=seq_len, n_pairs=n_pairs,
        vocab_size=vocab_size, device=device,
    )
    correct = total = 0
    for i in range(0, n_examples, batch_size):
        x = inputs[i:i + batch_size]
        y = labels[i:i + batch_size]
        logits, _ = model(x[:, :-1])
        target = y[:, :-1]
        mask = target != -1
        if mask.sum() == 0:
            continue
        preds = logits.argmax(dim=-1)
        correct += (preds[mask] == target[mask]).sum().item()
        total += int(mask.sum().item())

    acc = correct / total if total > 0 else float("nan")
    return {
        "seq_len": seq_len, "n_pairs": n_pairs, "vocab_size": vocab_size,
        "accuracy": acc, "correct": correct, "total": total,
        "n_examples": n_examples, "total_vocab": total_vocab,
    }


@torch.no_grad()
def mqar_grid(model, seq_lens=DEFAULT_SEQ_LENS, n_pairs_list=DEFAULT_N_PAIRS,
              vocab_size=DEFAULT_VOCAB_SIZE, n_examples=DEFAULT_N_EXAMPLES,
              batch_size=64, device="cpu", verbose=True):
    """Varre o MQAR em (seq_len × n_pairs). Retorna lista de células + meta."""
    cells, skipped = [], []
    for sl in seq_lens:
        for npr in n_pairs_list:
            cell = _eval_mqar_cell(model, sl, npr, vocab_size, n_examples,
                                   batch_size, device)
            if cell is None:
                skipped.append({"seq_len": sl, "n_pairs": npr,
                                "reason": "prefixo nao cabe em seq_len"})
                continue
            cells.append(cell)
            if verbose:
                print(f"  MQAR seq_len={sl:>5} n_pairs={npr:>3} | "
                      f"acc={cell['accuracy']:.4f} "
                      f"({cell['correct']}/{cell['total']})")
    return {
        "task": "mqar_grid",
        "chance_level": _chance_level(vocab_size),
        "vocab_size": vocab_size,
        "cells": cells,
        "skipped": skipped,
    }


@torch.no_grad()
def niah_sweep(model, seq_lens=DEFAULT_NIAH_SEQ_LENS, n_keys=8, n_queries=4,
               vocab_size=DEFAULT_VOCAB_SIZE, n_examples=200, batch_size=16,
               device="cpu", verbose=True):
    """RULER-NIAH em vários seq_len: recall com distância chave→query LONGA."""
    cells = []
    for sl in seq_lens:
        # NIAH exige seq_len >= 2*n_keys + bloco de query; pula se não couber.
        if sl < 2 * n_keys + (2 * min(n_queries, n_keys) + 1):
            continue
        out = evaluate_ruler(
            model, task="niah", batch_size=batch_size, device=device,
            n_examples=n_examples, seq_len=sl, n_keys=n_keys,
            n_queries=n_queries, vocab_size=vocab_size,
        )
        out["seq_len"] = sl
        cells.append(out)
        if verbose:
            print(f"  NIAH seq_len={sl:>5} | "
                  f"acc={out['ruler_niah_accuracy']:.4f} "
                  f"({out['correct']}/{out['total']})")
    return {
        "task": "niah_sweep",
        "chance_level": _chance_level(vocab_size),
        "vocab_size": vocab_size,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Runner por checkpoint (carrega o modelo e roda os dois sweeps)
# ---------------------------------------------------------------------------

def sweep_checkpoint(checkpoint_path, out_dir="results", device=None,
                     mqar_kwargs=None, niah_kwargs=None, save=True):
    """
    Carrega um checkpoint e roda o grid MQAR + sweep NIAH. Persiste um JSON
    nomeado pela variante e step. Importável e chamável inline no Colab.

    O nome dos parâmetros do mixer Mamba depende do backend (kernels: mixer.mamba.*;
    torch: mixer.mixer.*); load_state_dict quebra se o backend ATIVO divergir do
    backend com que o checkpoint foi salvo. Avisamos claramente nesse caso.
    """
    # Garante que a raiz do projeto está no path (p/ importar config/model).
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from config import TrainConfig  # noqa: F401 (mantém compat. de unpickle)
    from model import HybridModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ckpt["model_cfg"]
    step = ckpt.get("step", 0)
    pattern = model_cfg.block_pattern
    variant = f"{pattern.count('M')}:{pattern.count('A')}"

    ckpt_backend = ckpt.get("mamba_backend")
    cur_backend = os.environ.get("MAMBA_BACKEND", "torch")
    if "M" in pattern and ckpt_backend is not None and ckpt_backend != cur_backend:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} salvo com backend '{ckpt_backend}', "
            f"mas o ativo é '{cur_backend}'. Os pesos do mixer Mamba NÃO carregam "
            f"(nomes de parâmetros diferentes). Rode setup_env.setup() no mesmo "
            f"backend do treino antes de avaliar."
        )

    model = HybridModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(model.describe())

    t0 = time.time()
    mqar = mqar_grid(model, device=device, **(mqar_kwargs or {}))
    niah = niah_sweep(model, device=device, **(niah_kwargs or {}))
    dt = time.time() - t0

    results = {
        "checkpoint": checkpoint_path,
        "variant": variant,
        "step": step,
        "mamba_backend": ckpt_backend,
        "elapsed_s": round(dt, 1),
        "mqar_grid": mqar,
        "niah_sweep": niah,
    }

    if save:
        os.makedirs(out_dir, exist_ok=True)
        safe = variant.replace(":", "-")
        out_path = os.path.join(out_dir, f"recall_sweep_{safe}_step{step:07d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResultados salvos em: {out_path}")
        results["out_path"] = out_path

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Selftest (sem GPU): valida o GRID com o modelo-oráculo do MQAR.
# ---------------------------------------------------------------------------

def selftest() -> bool:
    """
    Reusa o oráculo do eval/mqar (lê o prefixo e responde 100%) para confirmar
    que o grid mede as posições certas em VÁRIOS (seq_len, n_pairs). O oráculo
    deve dar 1.0 em toda célula viável; células inviáveis devem ser puladas.
    """
    from eval.mqar import _OracleModel

    print("[selftest] grid MQAR com modelo-oráculo (deve dar 1.0 em toda célula)...")
    res = mqar_grid(
        _OracleModel(),
        seq_lens=(32, 64, 128),
        n_pairs_list=(4, 8, 16),
        vocab_size=64, n_examples=32, batch_size=16, device="cpu",
        verbose=True,
    )
    cells = res["cells"]
    ok = len(cells) > 0
    for c in cells:
        if not (c["total"] > 0 and abs(c["accuracy"] - 1.0) < 1e-9):
            print(f"  [FALHA] célula seq_len={c['seq_len']} n_pairs={c['n_pairs']} "
                  f"acc={c['accuracy']} (esperado 1.0)")
            ok = False
    # A célula (seq_len=32, n_pairs=16) deve ser PULADA: 2*16+1+2 = 35 > 32.
    skipped_pairs = {(s["seq_len"], s["n_pairs"]) for s in res["skipped"]}
    if (32, 16) not in skipped_pairs:
        print("  [FALHA] esperava pular a célula inviável (seq_len=32, n_pairs=16)")
        ok = False
    print(f"  chance_level reportado = {res['chance_level']:.5f} (esperado {1/64:.5f})")
    ok = ok and abs(res["chance_level"] - 1 / 64) < 1e-9
    print("  [OK] selftest passou" if ok else "  [FALHA] selftest falhou")
    return ok


# ---------------------------------------------------------------------------
# CLI fina
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Varredura de recall (MQAR grid + RULER-NIAH) sobre um checkpoint"
    )
    parser.add_argument("--checkpoint", help="Caminho do checkpoint .pt")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--n_examples", type=int, default=DEFAULT_N_EXAMPLES,
                        help="exemplos por célula do grid MQAR")
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.checkpoint:
        raise SystemExit("Forneça --checkpoint ou use --selftest.")

    sweep_checkpoint(
        args.checkpoint, out_dir=args.out_dir,
        mqar_kwargs={"n_examples": args.n_examples, "vocab_size": args.vocab_size},
        niah_kwargs={"vocab_size": args.vocab_size},
    )


if __name__ == "__main__":
    main()
