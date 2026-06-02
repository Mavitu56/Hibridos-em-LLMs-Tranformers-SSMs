"""
model.py — HybridModel: stack configurável de blocos GQA e Mamba-2

Uso:
    from config import ModelConfig, VARIANTS
    from model import HybridModel

    cfg = ModelConfig(block_pattern=VARIANTS["3:1"])
    model = HybridModel(cfg)
    logits, loss = model(input_ids, targets)
"""

import torch
import torch.nn as nn
from config import ModelConfig
from blocks import GQABlock, MambaBlock, RMSNorm


class HybridStack(nn.Module):
    """
    Sequência de blocos GQA e Mamba-2 definida pelo block_pattern.

    block_pattern: lista de 'A' (atenção) ou 'M' (Mamba)
    Exemplo: ['M','M','M','A','M','M','M','A','M','M','M','A'] → variante 3:1
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert len(cfg.block_pattern) == cfg.n_layers, (
            f"block_pattern tem {len(cfg.block_pattern)} elementos, "
            f"mas n_layers={cfg.n_layers}"
        )

        blocks = []
        for block_type in cfg.block_pattern:
            if block_type == 'A':
                blocks.append(GQABlock(cfg))
            elif block_type == 'M':
                blocks.append(MambaBlock(cfg))
            else:
                raise ValueError(f"Tipo de bloco inválido: '{block_type}'. Use 'A' ou 'M'.")

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class HybridModel(nn.Module):
    """
    Modelo de linguagem híbrido SSM-Transformer.

    Arquitetura:
        Embedding → HybridStack → RMSNorm → LM Head

    O LM Head compartilha pesos com o Embedding (weight tying),
    padrão desde o GPT-2 que reduz parâmetros sem perda de performance.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.stack      = HybridStack(cfg)
        self.norm_final = RMSNorm(cfg.d_model)
        self.lm_head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: embedding e lm_head compartilham a mesma matriz
        self.lm_head.weight = self.embedding.weight

        # Inicialização dos pesos (estilo nanoGPT)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T)
        targets: torch.Tensor = None,      # (B, T) — None durante geração
    ):
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, (
            f"Sequência de comprimento {T} excede max_seq_len={self.cfg.max_seq_len}"
        )

        x = self.embedding(input_ids)       # (B, T, D)
        x = self.stack(x)                   # (B, T, D)
        x = self.norm_final(x)              # (B, T, D)
        logits = self.lm_head(x)            # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def count_params(self, exclude_embedding: bool = True) -> int:
        """
        Conta parâmetros treináveis.
        exclude_embedding=True é o padrão da literatura para comparações
        (o tamanho do vocabulário varia entre experimentos e distorce a conta).
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embedding:
            emb = self.embedding.weight.numel()
            total -= emb
        return total

    def describe(self) -> str:
        """Resumo da arquitetura para registrar na metodologia."""
        pattern = self.cfg.block_pattern
        n_attn  = pattern.count('A')
        n_mamba = pattern.count('M')
        params  = self.count_params()

        lines = [
            f"Variante:       {n_mamba}:{n_attn} (Mamba:Atenção)",
            f"Padrão:         {''.join(pattern)}",
            f"Parâmetros:     {params / 1e6:.1f}M (sem embedding)",
            f"d_model:        {self.cfg.d_model}",
            f"n_layers:       {self.cfg.n_layers}",
            f"n_heads (GQA):  {self.cfg.n_heads} Q / {self.cfg.n_kv_heads} KV",
            f"d_state (SSM):  {self.cfg.d_state}",
        ]
        return "\n".join(lines)
