"""
config.py — Configurações do modelo híbrido SSM-Transformer
Decisões registradas na Fase 2 do plano de pesquisa.
"""

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class ModelConfig:
    # --- Arquitetura geral ---
    n_layers: int = 12          # Total de blocos (fixo em todos os experimentos)
    d_model: int = 512          # Dimensão oculta (ajustada por variante para paridade)
    d_ff: int = 2048            # Dimensão do MLP (tipicamente 4 * d_model)
    vocab_size: int = 50257     # GPT-2 tokenizer
    max_seq_len: int = 2048
    dropout: float = 0.0        # Sem dropout nos experimentos base

    # --- Bloco GQA (atenção) ---
    n_heads: int = 8            # Cabeças de query
    n_kv_heads: int = 2         # Cabeças de key/value (GQA: n_kv_heads < n_heads)

    # --- Bloco Mamba-2 ---
    d_state: int = 128          # Dimensão do estado SSM
    d_conv: int = 4             # Kernel da convolução local
    mamba_expand: int = 2       # Fator de expansão interno do Mamba-2

    # --- Proporção SSM/Atenção ---
    # Define a sequência de blocos: 'M' = Mamba, 'A' = Atenção
    # Exemplos de variantes do experimento:
    #   0:12  → ['A'] * 12
    #   3:1   → ['M','M','M','A'] repetido 3x
    #   5:1   → ['M','M','M','M','M','A'] repetido 2x
    #   7:1   → ['M','M','M','M','M','M','M','A'] + ['M','M','M','M']
    #   12:0  → ['M'] * 12
    block_pattern: List[Literal['M', 'A']] = field(
        default_factory=lambda: ['A'] * 12
    )


@dataclass
class TrainConfig:
    # --- Dataset ---
    dataset: str = "slimpajama"         # SlimPajama, subconjunto de 15-30B tokens
    batch_size: int = 32
    grad_accumulation_steps: int = 4    # Batch efetivo = 128
    max_tokens: int = int(15e9)         # 15B tokens para experimentos base

    # --- Otimizador ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- Scheduler (cosine com warmup) ---
    warmup_steps: int = 2000
    lr_decay_steps: int = 100_000

    # --- Logging e checkpoints ---
    eval_interval: int = 500
    log_interval: int = 10
    out_dir: str = "checkpoints"


# --- Variantes experimentais pré-definidas ---
# Proporção SSM:Atenção → padrão de blocos para 12 camadas

def make_pattern(n_mamba: int, n_attn: int, total: int = 12) -> List[str]:
    """
    Gera padrão intercalado de blocos M e A.
    Ex: make_pattern(3, 1, 12) → ['M','M','M','A','M','M','M','A','M','M','M','A']
    """
    if n_attn == 0:
        return ['M'] * total
    if n_mamba == 0:
        return ['A'] * total

    unit = ['M'] * n_mamba + ['A'] * n_attn
    pattern = (unit * (total // len(unit) + 1))[:total]
    return pattern


VARIANTS = {
    "0:12":  make_pattern(0,  12, 12),   # Transformer puro (baseline)
    "3:1":   make_pattern(3,   1, 12),   # Estilo Jamba
    "5:1":   make_pattern(5,   1, 12),   # Intermediário
    "7:1":   make_pattern(7,   1, 12),   # Estilo Nemotron-H (~8% atenção)
    "12:0":  make_pattern(12,  0, 12),   # SSM puro (baseline)
}
