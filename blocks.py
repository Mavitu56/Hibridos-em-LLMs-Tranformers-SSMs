"""
blocks.py — Blocos intercambiáveis GQA e Mamba-2
Implementação própria sobre kernels do mamba-ssm (Opção B do plano).

Cada bloco expõe a mesma interface:
    forward(x: Tensor[B, T, D]) -> Tensor[B, T, D]

Isso permite que o HybridStack monte qualquer sequência de blocos
sem modificar o código de treino.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# ---------------------------------------------------------------------------
# Utilitários compartilhados
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm — mais estável que LayerNorm em larga escala (usado no Mamba)."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class MLP(nn.Module):
    """MLP com ativação SwiGLU (padrão moderno, usado em Llama e Mamba-2)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate(x) * up(x), depois projeta de volta
        return self.down_proj(self.drop(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Bloco de Atenção com GQA
# ---------------------------------------------------------------------------

class GQABlock(nn.Module):
    """
    Grouped-Query Attention (Ainslie et al., 2023) + MLP.

    n_heads:    cabeças de Query
    n_kv_heads: cabeças de Key/Value (n_kv_heads <= n_heads)
                Se n_kv_heads == 1  → Multi-Query Attention (MQA)
                Se n_kv_heads == n_heads → Multi-Head Attention (MHA) padrão

    Durante os experimentos, usamos n_kv_heads=2, n_heads=8 (proporção 4:1).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model deve ser divisível por n_heads"
        assert cfg.n_heads % cfg.n_kv_heads == 0, "n_heads deve ser divisível por n_kv_heads"

        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.d_model // cfg.n_heads
        self.n_groups   = cfg.n_heads // cfg.n_kv_heads  # quantas queries por KV

        # Projeções — Q tem n_heads, K e V têm n_kv_heads (economia de memória)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp   = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop  = nn.Dropout(cfg.dropout)

        # Máscara causal pré-computada (evita recalcular a cada forward)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x = self.norm1(x)

        # Projeta Q, K, V e reorganiza para (B, heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expande K e V para combinar com o número de heads de Q (GQA)
        # (B, n_kv_heads, T, head_dim) → (B, n_heads, T, head_dim)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # Atenção escalada com máscara causal
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(~self.causal_mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        # Combina cabeças e projeta de volta
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)

        # Conexão residual + MLP
        x = residual + out
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Bloco Mamba-2
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    Wrapper sobre o Mamba-2 (mamba-ssm) com interface idêntica ao GQABlock.

    O bloco Mamba-2 já possui projeções internas (in_proj/out_proj),
    então não precisamos de MLP externo — mas adicionamos uma camada MLP
    separada para manter a paridade de parâmetros com o GQABlock.

    Nota sobre paridade: a dimensão d_model deve ser ajustada por variante
    para que o total de parâmetros seja equivalente entre variantes.
    Isso é feito em ModelConfig antes de instanciar o modelo.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # Import local para não quebrar se mamba-ssm não estiver instalado
        try:
            from mamba_ssm import Mamba2
        except ImportError:
            raise ImportError(
                "mamba-ssm não encontrado. Instale com:\n"
                "  pip install mamba-ssm --extra-index-url "
                "https://download.pytorch.org/whl/cu121"
            )

        self.mamba = Mamba2(
            d_model=cfg.d_model,
            d_state=cfg.d_state,       # Dimensão do estado SSM
            d_conv=cfg.d_conv,         # Kernel da convolução local
            expand=cfg.mamba_expand,   # Fator de expansão interno
        )

        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp   = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual + Mamba-2 + Residual + MLP (estilo pré-norm)
        x = x + self.mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
