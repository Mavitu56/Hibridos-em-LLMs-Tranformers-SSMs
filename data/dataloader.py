"""
data/dataloader.py — Carregamento do SlimPajama em streaming + tokenização GPT-2.

Dataset (D5 da spec): DKYoon/SlimPajama-6B (NÃO cerebras/SlimPajama-627B).
    - streaming=True: evita baixar ~14 GB, contorna o viewer e sobrevive a
      reconexões de sessão no Colab.
    - Tokenizer: GPT-2 BPE via tiktoken (linhagem nanoGPT, vocab 50257).
      # DECISÃO: trocar p/ GPT-NeoX se preferir — só ajustar vocab_size, que
      # não conta na paridade (D3 exclui embedding).
    - Packing: tokens concatenados em sequências contíguas de block_size, com
      EOT entre documentos. Targets = inputs deslocados em 1.
    - Split de validação FIXO e pequeno (held-out determinístico) para que a
      perplexidade seja comparável entre runs e variantes.

Interface pública (consumida por train.py e eval/): make_dataloader(...).
"""

from typing import Iterator, Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader
import tiktoken

from config import TrainConfig, ModelConfig


DATASET_NAME = "DKYoon/SlimPajama-6B"

# Tamanho do held-out de validação, em nº de sequências (block_size cada).
# Pequeno e fixo => perplexidade determinística e barata de avaliar.
VAL_NUM_SEQUENCES = 256


class SlimPajamaDataset(IterableDataset):
    """
    Itera DKYoon/SlimPajama-6B em streaming, tokeniza com GPT-2 e empacota em
    blocos contíguos de block_size (+1 para gerar o target deslocado).

    split:
        "train"      -> stream do split de treino, limitado por max_tokens.
        "validation" -> primeiras VAL_NUM_SEQUENCES sequências do split de
                        validação (held-out determinístico).
    """

    def __init__(self, train_cfg: TrainConfig, model_cfg: ModelConfig, split: str = "train"):
        super().__init__()
        self.block_size = train_cfg.block_size
        self.max_tokens = train_cfg.max_tokens
        self.split = split
        self.dataset_name = train_cfg.dataset or DATASET_NAME
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token

    def _hf_split(self) -> str:
        # DKYoon/SlimPajama-6B expõe os splits "train" e "validation".
        return "validation" if self.split == "validation" else "train"

    def _stream_tokens(self) -> Iterator[int]:
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            split=self._hf_split(),
            streaming=True,
        )

        # SHARDING POR WORKER (auditoria 2026-06-12): em IterableDataset cada
        # worker do DataLoader recebe uma CÓPIA do dataset e iteraria o stream
        # inteiro — com num_workers=2 cada sequência apareceria 2× no treino
        # (diversidade efetiva pela metade). Particionamos por documento:
        # o worker w processa os documentos i com i % num_workers == w.
        worker = torch.utils.data.get_worker_info()
        n_shards = worker.num_workers if worker is not None else 1
        shard_id = worker.id if worker is not None else 0

        # Limite de tokens: max_tokens no treino (dividido entre os shards);
        # o held-out de val é limitado por nº de sequências no __iter__.
        token_budget = (self.max_tokens // n_shards) if self.split == "train" else None

        yielded = 0
        for i, example in enumerate(ds):
            if i % n_shards != shard_id:
                continue
            text = example.get("text", "")
            if not text:
                continue
            ids = self.enc.encode_ordinary(text)
            ids.append(self.eot)  # separador de documentos
            for tok in ids:
                yield tok
                yielded += 1
                if token_budget is not None and yielded >= token_budget:
                    return

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        block = self.block_size + 1  # +1 p/ target deslocado
        buf = []
        seqs_emitted = 0
        max_seqs = VAL_NUM_SEQUENCES if self.split == "validation" else None

        for tok in self._stream_tokens():
            buf.append(tok)
            if len(buf) == block:
                chunk = torch.tensor(buf, dtype=torch.long)
                yield chunk[:-1], chunk[1:]   # (input, target)
                buf = []
                seqs_emitted += 1
                if max_seqs is not None and seqs_emitted >= max_seqs:
                    return


def make_dataloader(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    split: str = "train",
    num_workers: int = 0,
) -> DataLoader:
    """
    DataLoader sobre o SlimPajama em streaming.

    num_workers=0 por padrão (decisão 2026-06-17): com workers > 0 o DataLoader
    usa multiprocessing e cada worker reabre o stream HF no 1º next(); no Colab
    isso TRAVOU silenciosamente o hybrid_5_1 (15 min sem log nem uso de GPU,
    enquanto load_dataset/1º documento em processo único respondem em ~2s). Com
    num_workers=0 o stream roda no processo principal — confiável; o gargalo é a
    GPU (o tok/s das runs anteriores não era limitado por CPU). Para reativar o
    sharding por worker, passe num_workers>0 explicitamente (ver _stream_tokens).
    Validação sempre usa 0 (held-out determinístico). pin_memory acelera CPU->GPU.
    """
    if split == "validation":
        num_workers = 0

    dataset = SlimPajamaDataset(train_cfg, model_cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Held-out de validação em memória (cache de processo)
# ---------------------------------------------------------------------------
# O val fixo tem só VAL_NUM_SEQUENCES sequências (~2 MB em int64) — materializar
# uma vez evita reabrir o stream HF a cada avaliação (lento e sujeito a falha
# de rede no meio do treino, o que derrubaria a run inteira).

_VAL_CACHE: dict = {}


def get_val_batches(train_cfg: TrainConfig, model_cfg: ModelConfig) -> list:
    """Lista de (x, y) do held-out de validação, cacheada por (dataset, block, batch)."""
    key = (train_cfg.dataset, train_cfg.block_size, train_cfg.batch_size)
    if key not in _VAL_CACHE:
        loader = make_dataloader(train_cfg, model_cfg, split="validation", num_workers=0)
        _VAL_CACHE[key] = [(x, y) for x, y in loader]
    return _VAL_CACHE[key]
