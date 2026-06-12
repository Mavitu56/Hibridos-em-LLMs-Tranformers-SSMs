"""
blocks.py — Blocos intercambiáveis GQA e Mamba-2

Cada bloco expõe a MESMA interface pública (contrato INVARIANTE, §3 da spec):
    block = MambaBlock(cfg)   # recebe um ModelConfig
    out   = block(x)          # x: (B, T, D) -> out: (B, T, D), mesmo dtype/device

O mesmo vale para GQABlock. Isso permite que o HybridStack monte qualquer
sequência de blocos sem tocar no código de treino.

Backend do Mamba-2 (selecionado em runtime via os.environ["MAMBA_BACKEND"],
definido por setup_env.py):
    "kernels" -> mamba_ssm.Mamba2          (fast path CUDA)
    "torch"   -> transformers.Mamba2Mixer  (SSD puro PyTorch, fallback)
"""

import math
import os
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

def _rope_cos_sin(T: int, head_dim: int, theta: float, device, dtype):
    """cos/sin do RoPE para T posições, em float32 (estabilidade), cast no fim."""
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    ))
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                  # (T, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Aplica RoPE (convenção meia-rotação do Llama). x: (B, H, T, head_dim)."""
    x1, x2 = x.chunk(2, dim=-1)                       # (B, H, T, head_dim/2) cada
    # cos/sin: (T, head_dim/2) -> broadcast (1, 1, T, head_dim/2)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class GQABlock(nn.Module):
    """
    Grouped-Query Attention (Ainslie et al., 2023) + MLP.

    n_heads:    cabeças de Query
    n_kv_heads: cabeças de Key/Value (n_kv_heads <= n_heads)
                Se n_kv_heads == 1  → Multi-Query Attention (MQA)
                Se n_kv_heads == n_heads → Multi-Head Attention (MHA) padrão

    Durante os experimentos, usamos n_kv_heads=2, n_heads=8 (proporção 4:1).

    Posicional: RoPE (cfg.use_rope, default True) — sem parâmetros, logo não
    afeta a paridade D3. Com use_rope=False o bloco fica NoPE (regime Jamba).
    A atenção usa F.scaled_dot_product_attention (Flash/mem-efficient quando
    disponível): não materializa a matriz T×T — essencial p/ caber em T4 fp32.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model deve ser divisível por n_heads"
        assert cfg.n_heads % cfg.n_kv_heads == 0, "n_heads deve ser divisível por n_kv_heads"
        if cfg.use_rope:
            assert (cfg.d_model // cfg.n_heads) % 2 == 0, "RoPE exige head_dim par"

        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.d_model // cfg.n_heads
        self.n_groups   = cfg.n_heads // cfg.n_kv_heads  # quantas queries por KV
        self.use_rope   = cfg.use_rope
        self.rope_theta = cfg.rope_theta
        self.attn_dropout = cfg.dropout

        # Projeções — Q tem n_heads, K e V têm n_kv_heads (economia de memória)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp   = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x = self.norm1(x)

        # Projeta Q, K, V e reorganiza para (B, heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = _rope_cos_sin(T, self.head_dim, self.rope_theta, x.device, q.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        # Expande K e V para combinar com o número de heads de Q (GQA)
        # (B, n_kv_heads, T, head_dim) → (B, n_heads, T, head_dim)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # Atenção causal via SDPA (kernel Flash/mem-efficient quando disponível)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        # Combina cabeças e projeta de volta
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)

        # Conexão residual + MLP
        x = residual + out
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Bloco Mamba-2
# ---------------------------------------------------------------------------

def _select_mamba_backend() -> str:
    """
    Lê MAMBA_BACKEND do ambiente (definido por setup_env.setup()).
    Default "torch" se ausente — fallback seguro que não exige kernels CUDA.
    """
    backend = os.environ.get("MAMBA_BACKEND", "torch")
    if backend not in ("kernels", "torch"):
        raise ValueError(
            f"MAMBA_BACKEND='{backend}' inválido. Use 'kernels' ou 'torch'. "
            f"Rode setup_env.setup() primeiro."
        )
    return backend


class _Mamba2KernelMixer(nn.Module):
    """Mixer Mamba-2 via kernels CUDA (mamba_ssm.Mamba2). (B,T,D) -> (B,T,D)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from mamba_ssm import Mamba2  # import tardio: só quando o backend é kernels

        self.mamba = Mamba2(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.mamba_expand,
            headdim=cfg.headdim,
            chunk_size=cfg.chunk_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


class _Mamba2TorchMixer(nn.Module):
    """
    Mixer Mamba-2 via transformers.Mamba2Mixer (caminho torch_forward, puro PyTorch).

    A arquitetura SSD subjacente é IDÊNTICA à de Dao & Gu (2024); só muda a
    implementação (sem kernels CUDA customizados). Registrado no CHANGELOG por
    defensabilidade acadêmica. (B,T,D) -> (B,T,D).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
        from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

        d_inner = cfg.mamba_expand * cfg.d_model
        num_heads = d_inner // cfg.headdim

        m2cfg = Mamba2Config(
            hidden_size=cfg.d_model,
            state_size=cfg.d_state,
            conv_kernel=cfg.d_conv,
            expand=cfg.mamba_expand,
            head_dim=cfg.headdim,
            num_heads=num_heads,
            n_groups=1,
            chunk_size=cfg.chunk_size,
            use_conv_bias=True,
            use_bias=False,
            hidden_act="silu",
            rms_norm=True,
            layer_norm_epsilon=1e-5,
        )
        self.mixer = Mamba2Mixer(m2cfg, layer_idx=0)
        self._init_ssm_params(m2cfg, num_heads)

    def _init_ssm_params(self, m2cfg, num_heads: int):
        """
        Init explícito de A_log / dt_bias / D (auditoria 2026-06-12).

        Instanciar Mamba2Mixer DIRETO (fora de um Mamba2PreTrainedModel) pula o
        _init_weights do HF: em transformers 4.x o dt_bias fica em 1.0 (Δt ≈
        softplus(1) ≈ 1.31 — fora da faixa [0.001, 0.1] do paper, instável); em
        versões mais novas os três viram torch.empty (lixo de memória → nan
        imediato). Replicamos aqui o init oficial (igual ao do mamba_ssm.Mamba2,
        que o faz no próprio __init__ — mantém os DOIS backends consistentes).
        """
        dt_min = getattr(m2cfg, "time_step_min", 0.001)
        dt_max = getattr(m2cfg, "time_step_max", 0.1)
        dt_floor = getattr(m2cfg, "time_step_floor", 1e-4)
        with torch.no_grad():
            # A ∈ {1..H}: A_log = log(A) (decaimento real negativo via -exp(A_log))
            A = torch.arange(1, num_heads + 1, dtype=torch.float32)
            self.mixer.A_log.copy_(torch.log(A))
            self.mixer.D.fill_(1.0)
            # dt ~ LogUniform[dt_min, dt_max]; dt_bias = softplus^{-1}(dt)
            dt = torch.exp(
                torch.rand(num_heads)
                * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            ).clamp(min=dt_floor)
            self.mixer.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forçamos torch_forward para não depender da autodetecção do fast-path
        # CUDA (que exige os kernels que, neste backend, não estão presentes).
        out = self.mixer.torch_forward(x, cache_params=None, attention_mask=None)
        # Versões recentes retornam Tensor; algumas retornam tupla — normalizamos.
        if isinstance(out, tuple):
            out = out[0]
        return out


class MambaBlock(nn.Module):
    """
    Bloco Mamba-2 com interface idêntica ao GQABlock: (B,T,D) -> (B,T,D).

    Estrutura pré-norm: x + Mamba(norm1(x)) ; x + MLP(norm2(x)).
    O mixer interno é escolhido por MAMBA_BACKEND (kernels|torch). O MLP externo
    usa d_ff_mamba (menor que o d_ff da atenção) para igualar a contagem de
    parâmetros por bloco entre Mamba e GQA — ver decisão de paridade em config.py.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # Falha alto se a divisibilidade quebrar (em vez de bloco silenciosamente torto).
        d_inner = cfg.mamba_expand * cfg.d_model
        if d_inner % cfg.headdim != 0:
            raise ValueError(
                f"d_inner={d_inner} não divisível por headdim={cfg.headdim}"
            )

        backend = _select_mamba_backend()
        if backend == "kernels":
            self.mixer = _Mamba2KernelMixer(cfg)
        else:
            self.mixer = _Mamba2TorchMixer(cfg)
        self.backend = backend

        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp   = MLP(cfg.d_model, cfg.d_ff_mamba, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
