"""
benchmarks/mqar.py — Multi-Query Associative Recall (MQAR) sintético

Avalia se o modelo consegue recuperar valores associados a chaves em
sequências longas — benchmark sensível à capacidade de memória do SSM.

Referência: Arora et al. (2023), "Zoology: Measuring and Improving Recall
in Efficient Language Models."

Uso:
    python benchmarks/mqar.py --checkpoint checkpoints/3:1/step_10000.pt
"""

import argparse
import random
import torch
import torch.nn.functional as F


def generate_mqar_examples(
    n_examples: int = 1000,
    seq_len: int = 128,
    n_pairs: int = 8,
    vocab_size: int = 512,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gera exemplos MQAR: prefixo com n_pairs (chave, valor) + sufixo de queries.

    Formato da sequência:
        [k1, v1, k2, v2, ..., kN, vN, SEP, q1, q2, ..., qM]

    O modelo deve prever o valor correto após cada query.
    Retorna (input_ids, labels) onde labels=-1 exceto nas posições de resposta.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Tokens reservados: 0=PAD, 1=SEP
    KEY_OFFSET   = 2
    VALUE_OFFSET = KEY_OFFSET + vocab_size
    SEP_TOKEN    = 1

    # Garante que o vocabulário caiba nos tokens reservados + keys + values
    total_vocab = VALUE_OFFSET + vocab_size

    all_inputs = []
    all_labels = []

    for _ in range(n_examples):
        keys   = rng.sample(range(vocab_size), n_pairs)
        values = [rng.randint(0, vocab_size - 1) for _ in range(n_pairs)]
        kv     = dict(zip(keys, values))

        # Prefixo: sequência de pares (key_token, value_token)
        prefix = []
        for k, v in zip(keys, values):
            prefix.append(KEY_OFFSET + k)
            prefix.append(VALUE_OFFSET + v)

        prefix.append(SEP_TOKEN)

        # Calcula quantas queries cabem no seq_len restante
        remaining = seq_len - len(prefix)
        # Cada query ocupa 2 posições: [key_token, value_token (label)]
        n_queries = min(n_pairs, remaining // 2)
        query_keys = rng.sample(keys, n_queries)

        suffix       = []
        label_seq    = [-1] * len(prefix)  # prefixo não supervisionado

        for qk in query_keys:
            suffix.append(KEY_OFFSET + qk)
            label_seq.append(-1)           # posição da chave: não supervisionada
            suffix.append(VALUE_OFFSET + kv[qk])
            label_seq.append(VALUE_OFFSET + kv[qk])  # posição do valor: supervisionada

        seq = prefix + suffix

        # Padding ou truncamento para seq_len
        if len(seq) < seq_len:
            pad_len  = seq_len - len(seq)
            seq      = seq + [0] * pad_len
            label_seq = label_seq + [-1] * pad_len
        else:
            seq      = seq[:seq_len]
            label_seq = label_seq[:seq_len]

        all_inputs.append(seq)
        all_labels.append(label_seq)

    inputs = torch.tensor(all_inputs, dtype=torch.long, device=device)
    labels = torch.tensor(all_labels, dtype=torch.long, device=device)
    return inputs, labels, total_vocab


@torch.no_grad()
def evaluate_mqar(
    model,
    n_examples: int = 1000,
    seq_len: int = 128,
    n_pairs: int = 8,
    vocab_size: int = 512,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    """
    Avalia o modelo em accuracy@1 no benchmark MQAR.
    Retorna dict com accuracy e métricas auxiliares.
    """
    model.eval()
    inputs, labels, total_vocab = generate_mqar_examples(
        n_examples=n_examples,
        seq_len=seq_len,
        n_pairs=n_pairs,
        vocab_size=vocab_size,
        device=device,
    )

    correct = 0
    total   = 0

    for i in range(0, n_examples, batch_size):
        x = inputs[i : i + batch_size]
        y = labels[i : i + batch_size]

        # O modelo recebe tokens [0..T-1] e prevê [1..T]
        logits, _ = model(x[:, :-1])  # (B, T-1, vocab)
        target    = y[:, 1:]          # (B, T-1)

        mask = target != -1
        if mask.sum() == 0:
            continue

        preds = logits.argmax(dim=-1)  # (B, T-1)
        correct += (preds[mask] == target[mask]).sum().item()
        total   += mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return {
        "mqar_accuracy": accuracy,
        "mqar_correct":  correct,
        "mqar_total":    total,
        "seq_len":       seq_len,
        "n_pairs":       n_pairs,
        "vocab_size":    vocab_size,
    }


def main():
    parser = argparse.ArgumentParser(description="MQAR benchmark para modelos híbridos")
    parser.add_argument("--checkpoint", required=True, help="Caminho do checkpoint .pt")
    parser.add_argument("--n_examples", type=int, default=1000)
    parser.add_argument("--seq_len",    type=int, default=128)
    parser.add_argument("--n_pairs",    type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    import json
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import ModelConfig
    from model import HybridModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg  = ckpt["model_cfg"]
    model = HybridModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(model.describe())

    results = evaluate_mqar(
        model,
        n_examples=args.n_examples,
        seq_len=args.seq_len,
        n_pairs=args.n_pairs,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        device=device,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
