"""
plot_results.py — Figura principal do TCC: desempenho × proporção SSM:Atenção.

Lê os JSONs gerados por eval/recall_sweep.py (um por variante, em
_recall_results/) e produz uma figura de 4 painéis com BARRAS DE ERRO (dp das
seeds): PPL, MQAR-carga (n_pairs alto), MQAR-distância (gap, seq_len alto) e
NIAH. O eixo x é a fração de atenção do stack (0/12 … 12/12 blocos), ordenando
as 5 variantes do SSM puro ao Transformer puro — a variável independente do
estudo. Linha de acaso (1/vocab) marcada nos painéis de recall.

Só leitura/plotagem — não toca nos checkpoints nem na avaliação.

Uso (Colab, inline):
    import plot_results as P
    P.main("/content/drive/MyDrive/hybrid_ckpts/_recall_results")
Uso (CLI):
    python plot_results.py --results_dir .../_recall_results --out figura.png
"""

import argparse
import glob
import json
import os


# Marcadores reportados na figura (os mesmos do CHANGELOG/Resultados).
# Escolhemos a célula mais estressante viável de cada eixo.
MQAR_PACK_NPAIRS = 64     # carga associativa alta
MQAR_PACK_SEQLEN = 1024
MQAR_GAP_NPAIRS = 8       # poucos pares, distância longa isola o efeito de dist.
MQAR_GAP_SEQLEN = 1024
NIAH_SEQLEN = 1024


def _attn_fraction(variant: str) -> float:
    """variant 'M:A' (ex.: '10:2') -> fração de atenção A/(M+A)."""
    m, a = (int(x) for x in variant.split(":"))
    return a / (m + a)


def _find_cell(grid, seq_len, n_pairs):
    for c in grid["cells"]:
        if c["seq_len"] == seq_len and c["n_pairs"] == n_pairs:
            return c
    return None


def _niah_cell(sweep, seq_len):
    for c in sweep["cells"]:
        if c["seq_len"] == seq_len:
            return c
    return None


def load_results(results_dir):
    """Carrega todos os recall_sweep_*.json e devolve lista ordenada por fração de atenção."""
    paths = sorted(glob.glob(os.path.join(results_dir, "recall_sweep_*.json")))
    if not paths:
        raise SystemExit(f"Nenhum recall_sweep_*.json em {results_dir}")
    rows = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        variant = d["variant"]
        pack = _find_cell(d["mqar_grid_pack"], MQAR_PACK_SEQLEN, MQAR_PACK_NPAIRS)
        gap = _find_cell(d["mqar_grid_gap"], MQAR_GAP_SEQLEN, MQAR_GAP_NPAIRS)
        niah = _niah_cell(d["niah_sweep"], NIAH_SEQLEN)
        rows.append({
            "variant": variant,
            "attn_frac": _attn_fraction(variant),
            "ppl": d["perplexity"]["perplexity"] if d.get("perplexity") else None,
            "pack": (pack["accuracy"], pack["acc_std"]) if pack else (None, 0),
            "gap": (gap["accuracy"], gap["acc_std"]) if gap else (None, 0),
            "niah": (niah["ruler_niah_accuracy"], niah["acc_std"]) if niah else (None, 0),
            "chance": d["mqar_grid_pack"]["chance_level"],
        })
    rows.sort(key=lambda r: r["attn_frac"])
    return rows


def main(results_dir="results", out="figura_proporcao.png"):
    import matplotlib
    matplotlib.use("Agg")  # backend sem display (Colab/headless)
    import matplotlib.pyplot as plt

    rows = load_results(results_dir)
    x = [r["attn_frac"] for r in rows]
    labels = [r["variant"] for r in rows]
    chance = rows[0]["chance"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) Perplexidade — menor é melhor
    ax = axes[0, 0]
    ax.plot(x, [r["ppl"] for r in rows], "o-", color="#333")
    ax.set_title("(a) Perplexidade (val) — menor é melhor")
    ax.set_ylabel("perplexidade")
    ax.invert_yaxis()  # melhor (menor) para cima

    def panel(ax, key, title):
        y = [r[key][0] for r in rows]
        e = [r[key][1] for r in rows]
        ax.errorbar(x, y, yerr=e, fmt="o-", capsize=4, color="#1f77b4")
        ax.axhline(chance, ls="--", lw=1, color="gray", label=f"acaso ({chance:.3f})")
        ax.set_title(title)
        ax.set_ylabel("accuracy@1")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=8)

    panel(axes[0, 1], "pack", f"(b) MQAR carga (n_pairs={MQAR_PACK_NPAIRS})")
    panel(axes[1, 0], "gap", f"(c) MQAR distância (seq_len={MQAR_GAP_SEQLEN})")
    panel(axes[1, 1], "niah", f"(d) NIAH (seq_len={NIAH_SEQLEN})")

    for ax in axes.flat:
        ax.set_xlabel("fração de atenção no stack  (SSM puro → Transformer puro)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Trade-off memória↔atenção × proporção SSM:Atenção (~50M, 1.5B tokens, 5 seeds)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figura salva em: {out}")

    # Tabela-resumo no stdout (útil para conferência rápida).
    print("\nvariante     attn%   PPL    pack            gap             niah")
    for r in rows:
        print(f"  {r['variant']:6}  {r['attn_frac']*100:4.0f}  {r['ppl']:.2f}  "
              f"{r['pack'][0]:.3f}±{r['pack'][1]:.3f}  "
              f"{r['gap'][0]:.3f}±{r['gap'][1]:.3f}  "
              f"{r['niah'][0]:.3f}±{r['niah'][1]:.3f}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figura desempenho × proporção SSM:Atenção")
    parser.add_argument("--results_dir", default="results",
                        help="Diretório com recall_sweep_*.json")
    parser.add_argument("--out", default="figura_proporcao.png")
    args = parser.parse_args()
    main(args.results_dir, args.out)
