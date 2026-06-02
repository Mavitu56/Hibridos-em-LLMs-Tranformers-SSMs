"""
check_parity.py — Verifica contagem de parâmetros entre variantes

Execute ANTES de treinar qualquer variante para confirmar paridade.
Se os números divergirem muito (>5%), ajuste d_model em ModelConfig.

Uso:
    python check_parity.py
"""

import torch
from config import ModelConfig, VARIANTS
from model import HybridModel


def check_all_variants():
    print("=" * 55)
    print("Verificação de paridade de parâmetros")
    print("=" * 55)

    results = {}
    for name, pattern in VARIANTS.items():
        cfg = ModelConfig(block_pattern=pattern)
        model = HybridModel(cfg)
        params = model.count_params(exclude_embedding=True)
        results[name] = params
        desc = model.describe()
        print(f"\n--- Variante {name} ---")
        print(desc)

    print("\n" + "=" * 55)
    print("Resumo de parâmetros (sem embedding):")
    print("-" * 55)

    vals = list(results.values())
    ref  = vals[0]
    for name, params in results.items():
        diff = (params - ref) / ref * 100
        flag = "⚠️  DIVERGÊNCIA" if abs(diff) > 5 else "✓"
        print(f"  {name:8s}  {params/1e6:6.1f}M   ({diff:+.1f}%)  {flag}")

    print("=" * 55)

    max_diff = max(abs(p - ref) / ref for p in vals) * 100
    if max_diff > 5:
        print(f"\n⚠️  Divergência máxima: {max_diff:.1f}% — ajuste d_model antes de treinar.")
    else:
        print(f"\n✓ Paridade OK — divergência máxima: {max_diff:.1f}%")


if __name__ == "__main__":
    # Verifica se CUDA está disponível
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    check_all_variants()
