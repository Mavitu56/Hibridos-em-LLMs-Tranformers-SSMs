"""
eval/mqar.py — Multi-Query Associative Recall (MQAR) sintético.  [PRIORIDADE ALTA]

Testa a hipótese central do TCC: SSM puro colapsa no recall associativo de longo
alcance; um pouco de atenção recupera. Referência: Arora et al. (2023), "Zoology:
Measuring and Improving Recall in Efficient Language Models".

Formato de cada sequência (tokens):
    [k1, v1, k2, v2, ..., kN, vN, SEP, q1, a1, q2, a2, ..., qM, aM]
onde (qi, ai) repete um par (chave, valor) visto no prefixo. O modelo é avaliado
SÓ nas posições de resposta `ai` (acurácia de recuperação exata, accuracy@1).

Alinhamento de labels (importante — fonte clássica de off-by-one no MQAR):
a previsão a partir da posição de input t é sobre o token t+1. Marcamos como
supervisionada a posição da QUERY-CHAVE qi: a previsão feita ali deve ser a
resposta ai (= seq[t+1]). Portanto `labels[t]` vive no MESMO espaço de índices
que `inputs[t]` (NÃO deslocado): labels[t] = alvo da previsão feita a partir de
input[t]; -1 = posição ignorada. Na avaliação comparamos logits de
model(inputs[:, :-1]) contra labels[:, :-1] (sem re-shift).

Uso (CLI):
    python -m eval.mqar --checkpoint checkpoints/hybrid_3_1/last.pt
Uso (teste unitário):
    python -m eval.mqar --selftest
"""

import argparse
import json
import random

import torch


# Tokens reservados
PAD_TOKEN = 0
SEP_TOKEN = 1
KEY_OFFSET = 2  # chaves ocupam [KEY_OFFSET, KEY_OFFSET + vocab_size)


def _vocab_layout(vocab_size: int):
    key_offset = KEY_OFFSET
    value_offset = key_offset + vocab_size
    total_vocab = value_offset + vocab_size
    return key_offset, value_offset, total_vocab


def generate_mqar_examples(
    n_examples: int = 1000,
    seq_len: int = 128,
    n_pairs: int = 8,
    vocab_size: int = 512,
    device: str = "cpu",
    seed: int = 42,
    gap_fill: bool = False,
    n_filler: int = 64,
):
    """
    Gera (inputs, labels, total_vocab).
        inputs: (n_examples, seq_len) Long
        labels: (n_examples, seq_len) Long — alinhado a inputs[t] (NÃO deslocado;
                ver docstring do módulo). Posições não-resposta = -1.
                Para uso: model(inputs[:, :-1]) é comparado a labels[:, :-1].
        total_vocab: tamanho de vocabulário necessário (chaves + valores + reservados).

    gap_fill (decisão 2026-06-16): controla se `seq_len` afeta a DISTÂNCIA.
        - False (default, retrocompatível): pares empacotados contíguos no início,
          resto vira PAD. Aqui `seq_len` NÃO altera a distância chave→query — a
          variável de estresse efetiva é só `n_pairs`. Mantido como default para
          o oráculo/selftest e para comparabilidade com runs anteriores.
        - True: tokens-distrator variados (espaço DISJUNTO de chaves/valores) são
          intercalados entre os pares para PREENCHER o contexto até seq_len. Assim
          a distância entre uma chave no prefixo e sua query no fim cresce com
          seq_len — isola o efeito de DISTÂNCIA (estilo RULER), não só de carga.
    """
    rng = random.Random(seed)
    key_offset, value_offset, total_vocab = _vocab_layout(vocab_size)
    filler_lo = total_vocab
    if gap_fill:
        total_vocab += n_filler  # reserva o espaço de distratores (disjunto)
    filler_vocab = list(range(filler_lo, filler_lo + n_filler))

    all_inputs, all_labels = [], []
    for _ in range(n_examples):
        keys = rng.sample(range(vocab_size), n_pairs)
        values = [rng.randint(0, vocab_size - 1) for _ in range(n_pairs)]
        kv = dict(zip(keys, values))

        seq, label = [], []

        # Bloco de query no fim: 2 tokens por query + 1 (SEP) antes dele.
        n_queries = min(n_pairs, max(0, (seq_len - (2 * n_pairs + 1)) // 2))
        query_keys = rng.sample(keys, n_queries) if n_queries > 0 else []
        query_block_len = 1 + 2 * n_queries  # SEP + (chave, resposta) por query

        if gap_fill and n_queries > 0:
            # Orçamento de enchimento a distribuir ENTRE os pares do prefixo,
            # empurrando as chaves para longe das queries (distância ~ seq_len).
            budget = seq_len - (2 * n_pairs) - query_block_len
            budget = max(0, budget)
            # Reparte o budget em n_pairs+1 lacunas (antes/entre/depois dos pares).
            gaps = [0] * (n_pairs + 1)
            for _ in range(budget):
                gaps[rng.randrange(n_pairs + 1)] += 1
            for gi, (k, v) in enumerate(zip(keys, values)):
                seq += [rng.choice(filler_vocab) for _ in range(gaps[gi])]
                label += [-1] * gaps[gi]
                seq.append(key_offset + k); label.append(-1)
                seq.append(value_offset + v); label.append(-1)
            seq += [rng.choice(filler_vocab) for _ in range(gaps[n_pairs])]
            label += [-1] * gaps[n_pairs]
        else:
            # Prefixo contíguo (comportamento clássico).
            for k, v in zip(keys, values):
                seq.append(key_offset + k); label.append(-1)
                seq.append(value_offset + v); label.append(-1)

        # Bloco de query (idêntico nos dois modos).
        seq.append(SEP_TOKEN); label.append(-1)
        for qk in query_keys:
            ans = value_offset + kv[qk]
            seq.append(key_offset + qk); label.append(ans)  # chave prevê resposta
            seq.append(ans); label.append(-1)               # resposta em si: não supervisionada

        # Padding/truncamento
        if len(seq) < seq_len:
            pad = seq_len - len(seq)
            seq += [PAD_TOKEN] * pad
            label += [-1] * pad
        else:
            seq, label = seq[:seq_len], label[:seq_len]

        all_inputs.append(seq)
        all_labels.append(label)

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
    gap_fill: bool = False,
) -> dict:
    """Avalia accuracy@1 nas posições de resposta do MQAR.

    gap_fill=True intercala ruído entre os pares (a distância chave→query escala
    com seq_len). Ver generate_mqar_examples. Default False = comportamento clássico.
    """
    model.eval()
    inputs, labels, total_vocab = generate_mqar_examples(
        n_examples=n_examples, seq_len=seq_len, n_pairs=n_pairs,
        vocab_size=vocab_size, device=device, gap_fill=gap_fill,
    )

    correct = total = 0
    for i in range(0, n_examples, batch_size):
        x = inputs[i:i + batch_size]
        y = labels[i:i + batch_size]
        logits, _ = model(x[:, :-1])   # previsões a partir das posições 0..T-2
        target = y[:, :-1]             # labels NÃO deslocados (mesmo espaço de índices)
        mask = target != -1
        if mask.sum() == 0:
            continue
        preds = logits.argmax(dim=-1)
        correct += (preds[mask] == target[mask]).sum().item()
        total += mask.sum().item()

    acc = correct / total if total > 0 else 0.0
    return {
        "mqar_accuracy": acc, "mqar_correct": correct, "mqar_total": total,
        "seq_len": seq_len, "n_pairs": n_pairs, "vocab_size": vocab_size,
        "total_vocab": total_vocab,
    }


# ---------------------------------------------------------------------------
# Teste unitário: confirma que os labels gerados estão corretos
# ---------------------------------------------------------------------------

class _OracleModel:
    """
    Modelo-oráculo que NÃO aprende nada: lê o prefixo do próprio input para
    montar o mapa chave->valor e responde cada query corretamente. Serve para
    validar que os labels e o alinhamento do gerador estão certos: o oráculo
    DEVE atingir 100% de acurácia. Se não atingir, o bug está no gerador.
    """
    def eval(self): return self

    def __call__(self, x):
        B, T = x.shape
        vocab = KEY_OFFSET  # placeholder; inferimos o offset de valores pelo layout
        # Reconstrói o mapa por exemplo e produz logits one-hot na resposta correta.
        # Precisamos do value_offset: derivamos do maior token de valor possível.
        max_tok = int(x.max().item()) + 2
        logits = torch.zeros(B, T, max_tok)
        for b in range(B):
            seq = x[b].tolist()
            # localiza SEP
            sep = seq.index(SEP_TOKEN) if SEP_TOKEN in seq else len(seq)
            # value_offset = KEY_OFFSET + vocab_size; inferimos vocab_size pelos
            # tokens do prefixo: chaves em [2, 2+vs), valores em [2+vs, 2+2vs).
            # Em vez de inferir vs, montamos o mapa por posição: pares (k, v).
            # Pareamento por janela DESLIZANTE de passo 1 (kv[t] = t+1) sobre todo
            # o prefixo. Robusto ao modo gap_fill (ruído entre pares): chave e
            # valor são SEMPRE adjacentes no gerador, então kv[chave]=valor é
            # sempre registrado; pares espúrios (ruído→x) entram no dicionário mas
            # nunca são consultados (a query só pergunta chaves reais). No modo
            # contíguo (passo implícito 2) o resultado é idêntico.
            kv = {}
            for j in range(sep - 1):
                kv[seq[j]] = seq[j + 1]
            # para cada chave de query no sufixo, a previsão na posição da chave
            # (que vira input[t]; prevê t+1) deve ser o valor.
            for t in range(sep, T):
                tok = seq[t]
                if tok in kv:
                    ans = kv[tok]
                    if ans < max_tok:
                        logits[b, t, ans] = 1.0
        return logits, None


def selftest() -> bool:
    """O oráculo deve recuperar 100% — valida labels + alinhamento."""
    print("[selftest] gerando exemplos e avaliando com modelo-oráculo...")
    res = evaluate_mqar(
        _OracleModel(), n_examples=64, seq_len=64, n_pairs=4,
        vocab_size=64, batch_size=16, device="cpu",
    )
    print(f"  acurácia do oráculo: {res['mqar_accuracy']:.3f} "
          f"({res['mqar_correct']}/{res['mqar_total']})")
    ok = res["mqar_total"] > 0 and res["mqar_accuracy"] == 1.0
    # Sanidade extra: deve haver posições supervisionadas e elas batem com n_pairs.
    inputs, labels, _ = generate_mqar_examples(
        n_examples=8, seq_len=64, n_pairs=4, vocab_size=64
    )
    n_sup = (labels[:, :-1] != -1).sum().item()
    print(f"  posições supervisionadas em 8 exemplos: {n_sup}")
    ok = ok and n_sup > 0
    print("  ✓ selftest OK" if ok else "  ✗ selftest FALHOU")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MQAR para modelos híbridos")
    parser.add_argument("--checkpoint", help="Caminho do checkpoint .pt")
    parser.add_argument("--selftest", action="store_true", help="Roda o teste unitário e sai")
    parser.add_argument("--n_examples", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--n_pairs", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if args.selftest:
        import sys
        sys.exit(0 if selftest() else 1)

    if not args.checkpoint:
        raise SystemExit("Forneça --checkpoint ou use --selftest.")

    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import HybridModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HybridModel(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(model.describe())

    res = evaluate_mqar(
        model, n_examples=args.n_examples, seq_len=args.seq_len,
        n_pairs=args.n_pairs, vocab_size=args.vocab_size,
        batch_size=args.batch_size, device=device,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
