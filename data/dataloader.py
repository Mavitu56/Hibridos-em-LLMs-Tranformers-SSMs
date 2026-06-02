"""
data/dataloader.py — Carregamento do SlimPajama com tokenização GPT-2

Iterador infinito de batches (input_ids, targets) com targets shifted by 1.
Suporte a subconjunto controlado por max_tokens via TrainConfig.
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Iterator, Tuple
import tiktoken

from config import TrainConfig, ModelConfig


class SlimPajamaDataset(IterableDataset):
    """
    Itera sobre o SlimPajama (EleutherAI/slim_pajama-627B) em streaming,
    tokeniza com GPT-2 e fatia em blocos de max_seq_len tokens.

    Tokens máximos controlados por train_cfg.max_tokens para experimentos
    com orçamento computacional fixo.
    """

    def __init__(self, train_cfg: TrainConfig, model_cfg: ModelConfig, split: str = "train"):
        super().__init__()
        self.max_seq_len = model_cfg.max_seq_len
        self.max_tokens  = train_cfg.max_tokens
        self.split       = split
        self.enc         = tiktoken.get_encoding("gpt2")

    def _stream_tokens(self) -> Iterator[int]:
        """Abre o dataset em modo streaming e tokeniza documento por documento."""
        from datasets import load_dataset

        ds = load_dataset(
            "cerebras/SlimPajama-627B",
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        tokens_yielded = 0
        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            ids = self.enc.encode_ordinary(text)
            # Separador de documentos (EOT token do GPT-2)
            ids.append(self.enc.eot_token)
            for tok in ids:
                yield tok
                tokens_yielded += 1
                if tokens_yielded >= self.max_tokens:
                    return

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        buf = []
        block = self.max_seq_len + 1  # +1 para gerar targets com shift

        for tok in self._stream_tokens():
            buf.append(tok)
            if len(buf) == block:
                chunk = torch.tensor(buf, dtype=torch.long)
                yield chunk[:-1], chunk[1:]  # input, target
                buf = []


def make_dataloader(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    split: str = "train",
    num_workers: int = 2,
) -> DataLoader:
    """
    Retorna um DataLoader infinito (batch_size de train_cfg).
    pin_memory=True acelera a transferência CPU→GPU quando CUDA disponível.
    """
    dataset = SlimPajamaDataset(train_cfg, model_cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
