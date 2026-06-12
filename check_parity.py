"""
check_parity.py — Verifica a paridade de parâmetros ATIVOS entre as 5 variantes.

Regra (D3 da spec): ~50M parâmetros ativos (EXCLUINDO embedding), tolerância ±5%.
O script FALHA (exit != 0) se qualquer variante sair da banda — assim ele serve
de gate executável antes de treinar.

Uso:
    python check_parity.py
"""

import sys

import torch

from config import ModelConfig, VARIANTS
from model import HybridModel


TARGET_PARAMS = 50e6     # alvo absoluto de parâmetros ativos
TOLERANCE = 0.05         # ±5%


def check_all_variants() -> bool:
    """Imprime a tabela de contagens e devolve True se TODAS passarem ±5%."""
    print("=" * 64)
    print(f"Paridade de parâmetros ativos — alvo {TARGET_PARAMS/1e6:.0f}M ±{TOLERANCE*100:.0f}%")
    print("=" * 64)

    results = {}
    for name, pattern in VARIANTS.items():
        cfg = ModelConfig(block_pattern=pattern)
        model = HybridModel(cfg)
        params = model.count_params(exclude_embedding=True)
        results[name] = params
        print(f"\n--- {name} ---")
        print(model.describe())
        del model  # libera memória ao varrer as 5 variantes

    print("\n" + "=" * 64)
    print(f"{'variante':12s} {'ativos':>10s} {'vs alvo':>9s}  status")
    print("-" * 64)

    lo = TARGET_PARAMS * (1 - TOLERANCE)
    hi = TARGET_PARAMS * (1 + TOLERANCE)
    all_ok = True
    for name, params in results.items():
        diff = (params - TARGET_PARAMS) / TARGET_PARAMS * 100
        ok = lo <= params <= hi
        all_ok = all_ok and ok
        flag = "OK" if ok else "FORA DA BANDA"
        print(f"{name:12s} {params/1e6:9.2f}M {diff:+8.1f}%  {flag}")

    # Divergência relativa entre variantes (informativa)
    vals = list(results.values())
    spread = (max(vals) - min(vals)) / min(vals) * 100
    print("-" * 64)
    print(f"Spread entre variantes (max-min)/min: {spread:.1f}%")
    print("=" * 64)

    if all_ok:
        print("\n✓ Paridade OK — todas as variantes dentro de ±5% de 50M ativos.")
    else:
        print(
            "\n✗ Paridade FALHOU. Ajuste d_ff / d_ff_mamba / mamba_expand em "
            "ModelConfig (config.py) e rode novamente."
        )
    return all_ok


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  (contagem de params independe do device)\n")
    ok = check_all_variants()
    sys.exit(0 if ok else 1)
