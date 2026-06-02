# CLAUDE.md — Contexto do projeto hybrid-ssm-lm

## O que é este projeto

TCC + artigo de revisão narrativa comparando 5 variantes arquiteturais de
modelos híbridos SSM-Transformer em escala controlada (~50M parâmetros).
O objetivo é medir o trade-off entre capacidade de memória sequencial (SSM)
e atenção global (Transformer) em tarefas de linguagem.

## As 5 variantes experimentais

| Variante | Padrão (12 blocos)              | Referência arquitetural |
|----------|---------------------------------|-------------------------|
| `0:12`   | AAAAAAAAAAAA                    | Transformer puro (GPT)  |
| `3:1`    | MMMAMMMAMMMА                    | Jamba (AI21 Labs)       |
| `5:1`    | MMMMMAMMMMMА                    | Intermediário           |
| `7:1`    | MMMMMMMAMMMM                    | Nemotron-H (~8% atenção)|
| `12:0`   | MMMMMMMMMMMM                    | SSM puro (Mamba-2)      |

- **M** = bloco Mamba-2 (SSM com estado recorrente)
- **A** = bloco GQA (Grouped-Query Attention, n_heads=8, n_kv_heads=2)

## Regras de desenvolvimento

1. **Sempre rodar `check_parity.py` antes de treinar uma nova variante.**
   Paridade de parâmetros (±5%) entre variantes é pré-requisito para
   comparações justas. Se divergir, ajuste `d_model` em `ModelConfig`.

2. **Não modificar a interface pública de `config.py`, `blocks.py` e `model.py`.**
   Esses arquivos definem o contrato entre componentes. Mudanças internas
   são aceitas, mas as assinaturas públicas devem permanecer estáveis.

3. **Registrar decisões de implementação como comentários no código.**
   Este é código de pesquisa — legibilidade e rastreabilidade importam mais
   que otimização prematura.

## Fluxo de trabalho padrão

```bash
# 1. Verificar paridade de parâmetros
python check_parity.py

# 2. Treinar uma variante
python train.py --variant 3:1 --out_dir checkpoints/3:1 --max_steps 100000

# 3. Avaliar o checkpoint
python evaluate.py \
    --checkpoint checkpoints/3:1/step_0100000.pt \
    --benchmarks perplexity,lambada,hellaswag

# 4. Benchmark MQAR (memória associativa)
python benchmarks/mqar.py --checkpoint checkpoints/3:1/step_0100000.pt
```

## Estrutura de arquivos

```
hybrid-ssm-lm/
├── config.py           # ModelConfig, TrainConfig, VARIANTS, make_pattern()
├── blocks.py           # GQABlock, MambaBlock, RMSNorm, MLP
├── model.py            # HybridModel, HybridStack
├── check_parity.py     # Verificação de paridade de parâmetros
├── train.py            # Loop de treino (nanoGPT-style)
├── evaluate.py         # Perplexidade, LAMBADA, HellaSwag
├── data/
│   └── dataloader.py   # SlimPajamaDataset + make_dataloader()
├── benchmarks/
│   └── mqar.py         # Multi-Query Associative Recall sintético
├── requirements.txt
└── CLAUDE.md
```

## Dependências e ambiente

- Python 3.10+, PyTorch 2.x, CUDA 12.x
- `mamba-ssm>=2.0.0` requer GPU CUDA (sem suporte CPU)
- Instalar: `pip install -r requirements.txt`
- Para mamba-ssm em CUDA 12.1:
  `pip install mamba-ssm --extra-index-url https://download.pytorch.org/whl/cu121`

## Decisões de design registradas

- **Weight tying**: embedding e lm_head compartilham pesos (padrão GPT-2)
- **RMSNorm**: preferido sobre LayerNorm por estabilidade em larga escala
- **SwiGLU**: ativação do MLP em ambos os blocos (padrão Llama/Mamba-2)
- **GQA**: n_kv_heads=2, n_heads=8 — reduz memória KV cache sem perda significativa
- **Sem frameworks de alto nível**: loop explícito para rastreabilidade de pesquisa
- **bfloat16 autocast**: ativado automaticamente quando CUDA disponível
