# CLAUDE.md — Contexto do projeto hybrid-ssm-lm

## O que é este projeto

TCC + artigo de revisão narrativa comparando 5 variantes arquiteturais de
modelos híbridos SSM-Transformer em escala controlada (~50M parâmetros **ativos**,
excluindo embedding). O objetivo é medir o trade-off entre capacidade de memória
sequencial (SSM) e atenção global (Transformer) em tarefas de linguagem.

Ambiente de execução real: **Google Colab Pro** (GPU variável T4/L4/A100). Veja
`run_colab.ipynb` para o gate da Fase A e `CHANGELOG.md` para o histórico de
decisões (insumo da metodologia).

## As 5 variantes experimentais

| Variante     | Razão | Padrão (12 blocos) | Referência               |
|--------------|-------|--------------------|--------------------------|
| `attn_only`  | 0:12  | `AAAAAAAAAAAA`     | Transformer puro (GPT)   |
| `hybrid_3_1` | 3:1   | `MMMAMMMAMMMA`     | Jamba (AI21 Labs)        |
| `hybrid_5_1` | 5:1   | `MMMMMAMMMMMA`     | Intermediário            |
| `hybrid_7_1` | 7:1   | `MMMMMMMAMMMM`     | Nemotron-H (~8% atenção) |
| `ssm_only`   | 12:0  | `MMMMMMMMMMMM`     | SSM puro (Mamba-2)       |

- **M** = bloco Mamba-2 (SSM com estado recorrente)
- **A** = bloco GQA (n_heads=8, n_kv_heads=2)
- Nomes e razões são intercambiáveis via `resolve_variant()` em `config.py`.

## Regras de desenvolvimento

1. **Rodar `check_parity.py` antes de treinar.** Paridade de ~50M ativos (±5%)
   é pré-requisito; o script falha (exit≠0) se algo sair da banda. Ajuste fino
   por `d_ff_mamba` / `d_ff` (não por `d_model`, que é compartilhado).
2. **Não modificar a interface pública** de `config.py`, `blocks.py`, `model.py`.
   O contrato dos blocos é `(B,T,D) -> (B,T,D)`.
3. **Registrar decisões como comentários no código** e no `CHANGELOG.md`.

## Fluxo de trabalho padrão (Colab, inline — sem `!python`)

```python
import setup_env;            backend = setup_env.setup()   # kernels|torch
import run_experiments as R; R.phase_a(out_root="/content/drive/MyDrive/ckpts")
# ... gate passou ...
R.phase_b("/content/drive/MyDrive/ckpts")                  # hybrid_3_1 + MQAR/ppl
```

Treino/avaliação individuais:

```python
from train import train
from config import TrainConfig
train("hybrid_3_1", TrainConfig(out_dir="/content/drive/MyDrive/ckpts/hybrid_3_1"))
```

```bash
python evaluate.py --checkpoint .../last.pt --benchmarks perplexity,mqar,lambada
python -m eval.mqar --selftest      # teste unitário do gerador MQAR
```

## Estrutura de arquivos

```
hybrid-ssm-lm/
├── config.py           # ModelConfig, TrainConfig, make_pattern(), VARIANTS, resolve_variant()
├── blocks.py           # GQABlock, MambaBlock (switch de backend), RMSNorm, MLP
├── model.py            # HybridModel, HybridStack (init escalado por profundidade)
├── setup_env.py        # Instala kernels Mamba-2 por wheel OU fallback torch (MAMBA_BACKEND)
├── check_parity.py     # Paridade ±5% de 50M ativos (falha alto)
├── train.py            # train(variant, cfg)->metrics; resume no Drive, bf16/fp32
├── evaluate.py         # Dispatcher fino sobre eval/
├── run_experiments.py  # Orquestração Fase A (gate) / B / C
├── run_colab.ipynb     # Notebook do gate da Fase A
├── data/
│   └── dataloader.py   # DKYoon/SlimPajama-6B streaming, val determinístico
├── eval/
│   ├── mqar.py         # Multi-Query Associative Recall + selftest (prioridade alta)
│   ├── perplexity.py   # Perplexidade no val fixo
│   ├── lambada.py      # LAMBADA zero-shot (secundário)
│   ├── hellaswag.py    # HellaSwag zero-shot (secundário)
│   └── ruler.py        # Subconjunto RULER: NIAH + variable tracking (último)
├── benchmarks/mqar.py  # SHIM -> eval/mqar.py (compat.)
├── requirements.txt
├── CHANGELOG.md
└── CLAUDE.md
```

## Dependências e ambiente

- Python 3.10+, PyTorch 2.x, CUDA 12.x. Deps base em `requirements.txt`.
- **Kernels Mamba-2 não vão no requirements:** `setup_env.py` os instala por
  wheel pré-compilada (casando torch/cuda/cxx11abi/python) ou cai para o backend
  PyTorch puro (`transformers.Mamba2`). Nunca compilar do zero no Colab.

## Decisões de design registradas

- **Backend Mamba-2 selecionável** por `MAMBA_BACKEND` (kernels|torch); a SSD é a
  mesma de Dao & Gu (2024) nos dois casos. Wheels p/ o Colab atual (py3.12,
  torch 2.8, cu12) existem desde jan/2026 — backend kernels é viável de novo.
- **Init explícito de `A_log`/`dt_bias`/`D`** no backend torch (Mamba2Mixer fora
  de um PreTrainedModel não passa pelo `_init_weights` do HF) — mantém os dois
  backends consistentes e evita Δt fora da faixa [0.001, 0.1].
- **RoPE na atenção** (`use_rope=True`, sem parâmetros — não afeta paridade);
  `use_rope=False` reproduz o regime NoPE estilo Jamba. Atenção via
  `F.scaled_dot_product_attention` (Flash; sem matriz T×T explícita).
- **Sharding por worker no dataloader** (IterableDataset; evita stream duplicado)
  e **held-out de validação cacheado em memória** (`get_val_batches`).
- **Micro-batch adaptativo no backend torch** (`mamba_torch_microbatch=4`): o
  torch_forward do Mamba2Mixer aloca ~1 GiB/unidade de batch em T=1024 (OOM em
  A100 com B=16); train.py reduz batch e multiplica grad_accum — tokens/step
  idêntico. setup_env tenta wheels de minors anteriores do torch se o runtime
  for mais novo que a última wheel (Colab jun/2026: torch 2.11 vs wheels ≤2.10).
- **Paridade por `d_ff` dual:** `d_ff` (atenção) e `d_ff_mamba` (menor) igualam a
  contagem por bloco mantendo `d_model` compartilhado.
- **Init escalado por profundidade** (`1/sqrt(2·n_layers)`) nas projeções
  residuais de saída — principal medida anti-`nan`.
- **bf16 autocast** (fp32 + aviso em T4; nunca fp16 puro em Mamba).
- **Weight tying**, **RMSNorm**, **SwiGLU**, **GQA** (n_kv_heads=2).
- **Resume no Drive** com escrita atômica — assume queda de sessão no Colab.
- **Sem frameworks de alto nível**: loop explícito para rastreabilidade.
