"""
config.py — Configurações do modelo híbrido SSM-Transformer

Interface pública (NÃO alterar assinaturas — contrato com blocks.py/model.py):
    ModelConfig, TrainConfig, make_pattern(), VARIANTS

Decisões registradas no CHANGELOG.md (seção "config").
"""

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class ModelConfig:
    # --- Arquitetura geral ---
    n_layers: int = 12          # Total de blocos (fixo em todos os experimentos)
    d_model: int = 512          # Dimensão oculta — COMPARTILHADA por todos os blocos
                                # do stack (mexer aqui afeta GQA e Mamba de uma vez).
    vocab_size: int = 50257     # GPT-2 BPE (tiktoken). Não conta na paridade (D3).
    max_seq_len: int = 2048     # Capacidade máxima do modelo (RULER/long-context na Fase C).
    dropout: float = 0.0        # Sem dropout nos experimentos base.

    # --- MLP (SwiGLU) ---
    # DECISÃO DE PARIDADE: o bloco Mamba-2 carrega overhead fixo (in_proj/out_proj/conv),
    # então usamos d_ff distintos por tipo de bloco para igualar a contagem de parâmetros
    # POR BLOCO mantendo d_model compartilhado. Calibrado para ~50M ativos, maxdiff ~0.4%
    # entre as 5 variantes (ver check_parity.py). Ajuste fino via check_parity.
    d_ff: int = 2304            # d_ff dos blocos de ATENÇÃO (GQA).
    d_ff_mamba: int = 1600      # d_ff dos blocos MAMBA (menor p/ compensar o overhead do SSM).

    # --- Bloco GQA (atenção) ---
    n_heads: int = 8            # Cabeças de query
    n_kv_heads: int = 2         # Cabeças de key/value (GQA: n_kv_heads < n_heads)
    # RoPE nas cabeças de atenção (sem parâmetros => não afeta a paridade D3).
    # DECISÃO (auditoria 2026-06-12): Jamba/Nemotron-H treinam SEM posicional
    # explícito porque os blocos Mamba fornecem ordem; mas aqui o attn_only puro
    # ficaria NoPE — baseline enfraquecida e confundidor da variável proporção.
    # use_rope=False reproduz o regime NoPE (estilo Jamba) se desejado.
    use_rope: bool = True
    rope_theta: float = 10000.0

    # --- Bloco Mamba-2 ---
    d_state: int = 128          # Dimensão do estado SSM
    d_conv: int = 4             # Kernel da convolução local (curta)
    mamba_expand: int = 2       # Fator de expansão interno: d_inner = expand * d_model
    headdim: int = 64           # Dimensão por cabeça do SSM. d_inner DEVE ser divisível.
    chunk_size: int = 256       # Tamanho do chunk do algoritmo SSD (Dao & Gu, 2024)

    # --- Proporção SSM/Atenção ---
    # Sequência de blocos: 'M' = Mamba-2, 'A' = Atenção GQA. Gerada por make_pattern().
    block_pattern: List[Literal['M', 'A']] = field(
        default_factory=lambda: ['A'] * 12
    )

    def __post_init__(self):
        # Falha alto se a divisibilidade do Mamba-2 quebrar — evita bloco "torto"
        # silencioso (exigência da Tarefa 2 da spec).
        d_inner = self.mamba_expand * self.d_model
        if d_inner % self.headdim != 0:
            raise ValueError(
                f"d_inner = expand*d_model = {self.mamba_expand}*{self.d_model} = {d_inner} "
                f"não é divisível por headdim={self.headdim}. "
                f"Ajuste d_model, mamba_expand ou headdim em ModelConfig."
            )


@dataclass
class TrainConfig:
    # --- Dataset ---
    dataset: str = "DKYoon/SlimPajama-6B"   # D5: NÃO usar cerebras/SlimPajama-627B
    block_size: int = 1024                  # Comprimento de sequência no TREINO
    batch_size: int = 16                    # Micro-batch por step
    grad_accumulation_steps: int = 8        # Batch efetivo = 128 sequências

    # Micro-batch máximo quando MAMBA_BACKEND=torch e a variante tem blocos M.
    # O torch_forward do Mamba2Mixer materializa um tensor O(B·T·chunk·h·p) em
    # fp32 (~1 GiB por unidade de batch em T=1024): B=16 deu OOM até em A100
    # 40GB (Fase A, 2026-06-12). train.py reduz batch_size até este teto e
    # multiplica grad_accumulation_steps na mesma proporção — tokens/step e
    # batch efetivo IDÊNTICOS (comparabilidade preservada). 0 = desativa.
    mamba_torch_microbatch: int = 4

    # --- Orçamento de tokens (controla o "mesmo orçamento" entre variantes) ---
    # ~1.5B tokens: viável em Colab Pro, suficiente para perplexidade interpretável
    # e contraste no MQAR. max_steps deriva disto se não for passado explicitamente.
    max_tokens: int = int(1.5e9)
    max_steps: int = 0                      # 0 => derivar de max_tokens / tokens_por_step

    # --- Otimizador ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- Scheduler (cosine com warmup) ---
    warmup_frac: float = 0.05               # 5% dos steps em warmup linear
    min_lr_frac: float = 0.1                # piso = 10% da LR máxima

    # --- Logging, avaliação e checkpoints ---
    eval_interval: int = 1000               # steps entre avaliações de val/perplexidade
    eval_batches: int = 50                  # batches usados na perplexidade de val
    checkpoint_interval: int = 1000         # steps entre checkpoints (resume no Drive)
    log_interval: int = 20                  # steps entre logs de train loss
    out_dir: str = "checkpoints"            # raiz dos checkpoints (apontar p/ Drive no Colab)

    # --- Precisão ---
    # bf16 preferido; em GPU sem suporte (T4) o train.py cai para fp32 e avisa.
    # NUNCA fp16 puro em Mamba (instabilidade numérica conhecida).
    dtype: str = "bfloat16"


# ---------------------------------------------------------------------------
# Geração de padrões de blocos
# ---------------------------------------------------------------------------

def make_pattern(n_mamba: int, n_attn: int, total: int = 12) -> List[str]:
    """
    Gera o padrão de 12 blocos para uma razão SSM:Atenção.

    A unidade (n_mamba M seguidos de n_attn A) é repetida e truncada em `total`.
    Garante EXATAMENTE `total` blocos.

    Exemplos (total=12):
        make_pattern(0, 12) -> AAAAAAAAAAAA          (0:12, ~0% Mamba)
        make_pattern(3, 1)  -> MMMAMMMAMMMA          (3:1, 25% atenção, estilo Jamba)
        make_pattern(5, 1)  -> MMMMMAMMMMMA          (5:1, ~17% atenção)
        make_pattern(7, 1)  -> MMMMMMMAMMMM          (7:1, ~8% atenção, estilo Nemotron-H)
        make_pattern(12, 0) -> MMMMMMMMMMMM          (12:0, SSM puro)
    """
    if n_attn == 0:
        return ['M'] * total
    if n_mamba == 0:
        return ['A'] * total

    unit = ['M'] * n_mamba + ['A'] * n_attn
    pattern = (unit * (total // len(unit) + 1))[:total]
    return pattern


# ---------------------------------------------------------------------------
# Variantes experimentais (D2: variável independente = razão SSM:Atenção)
# ---------------------------------------------------------------------------
# Chaves nomeadas conforme a spec (§2). Os aliases "M:A" também são aceitos
# por quem prefere a notação numérica.

VARIANTS = {
    "attn_only":  make_pattern(0,  12, 12),   # 0:12  — Transformer puro (baseline)
    "hybrid_3_1": make_pattern(3,   1, 12),   # 3:1   — estilo Jamba
    "hybrid_5_1": make_pattern(5,   1, 12),   # 5:1   — intermediário
    "hybrid_7_1": make_pattern(7,   1, 12),   # 7:1   — estilo Nemotron-H (~8% atenção)
    "ssm_only":   make_pattern(12,  0, 12),   # 12:0  — SSM puro (baseline)
}

# Aliases de conveniência por razão numérica (compat. com fluxo antigo do CLAUDE.md).
VARIANT_ALIASES = {
    "0:12": "attn_only",
    "3:1":  "hybrid_3_1",
    "5:1":  "hybrid_5_1",
    "7:1":  "hybrid_7_1",
    "12:0": "ssm_only",
}


def resolve_variant(name: str) -> List[str]:
    """Aceita tanto o nome da spec ('hybrid_3_1') quanto a razão ('3:1')."""
    if name in VARIANTS:
        return VARIANTS[name]
    if name in VARIANT_ALIASES:
        return VARIANTS[VARIANT_ALIASES[name]]
    raise ValueError(
        f"Variante '{name}' desconhecida. "
        f"Opções: {list(VARIANTS.keys())} ou {list(VARIANT_ALIASES.keys())}"
    )
