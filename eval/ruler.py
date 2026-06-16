"""
eval/ruler.py — Subconjunto sintético do RULER (último na prioridade, Fase C).

Tarefas implementadas (token-level, self-contained, sem dependências externas):
  - Multi-key NIAH (Needle-In-A-Haystack): insere vários pares (chave, valor)
    ("agulhas") em meio a tokens de enchimento ("palheiro") e pergunta o valor
    de chaves específicas no fim. Estressa recall em contexto LONGO.
  - Variable Tracking: cadeia de atribuições v_i = v_{i-1} (ou valor inicial);
    o modelo deve reportar o valor final de uma variável após N saltos.

Ambas reusam o layout de tokens do MQAR (reservados + chaves + valores) e a
mesma convenção de alinhamento de labels (labels[t] no espaço de inputs[t];
avaliar logits de model(x[:,:-1]) contra labels[:,:-1]).

Uso:
    python -m eval.ruler --checkpoint ckpt.pt --task niah --seq_len 1024
    python -m eval.ruler --selftest
"""

import argparse
import json
import random

import torch

PAD, SEP, KO = 0, 1, 2


def _layout(vocab_size):
    return KO, KO + vocab_size, KO + 2 * vocab_size  # key_off, val_off, total


def generate_niah(n_examples=200, seq_len=1024, n_keys=8, n_queries=4,
                  vocab_size=512, device="cpu", seed=42, n_filler=64):
    """Multi-key NIAH: agulhas (k,v) espalhadas em enchimento; consulta no fim.

    Enchimento = RUÍDO VARIADO (auditoria 2026-06-16). Antes o palheiro era um
    ÚNICO token `FILLER` repetido por todo o haystack (~80% da sequência) — uma
    distribuição que nenhum LM treinado em texto natural vê e que é ADVERSÁRIA à
    atenção: uma sequência de tokens idênticos colapsa o padrão causal (todas as
    chaves/valores iguais → a query não destaca a agulha no mar de FILLER). Isso
    fazia o `attn_only` zerar no NIAH (0/800) enquanto Mamba/híbrido pontuavam —
    paradoxal e contrário à literatura. O RULER real (NVIDIA, Hsieh et al. 2024)
    usa ruído VARIADO (frases de ruído ou ensaios), não um token único. Aqui o
    palheiro é amostrado de um vocabulário de `n_filler` tokens-distrator
    DISJUNTO do espaço de chaves/valores (não há colisão possível com as agulhas).
    """
    rng = random.Random(seed)
    ko, vo, total = _layout(vocab_size)
    # Vocabulário de distratores: `n_filler` tokens reservados FORA de
    # [chaves, valores]. Disjunto → o modelo nunca confunde ruído com agulha.
    filler_lo = total
    filler_vocab = list(range(filler_lo, filler_lo + n_filler))
    total += n_filler

    X, L = [], []
    for _ in range(n_examples):
        keys = rng.sample(range(vocab_size), n_keys)
        vals = [rng.randint(0, vocab_size - 1) for _ in range(n_keys)]
        kv = dict(zip(keys, vals))

        # reserva espaço para o bloco de query no fim
        q_keys = rng.sample(keys, min(n_queries, n_keys))
        query_len = 2 * len(q_keys) + 1  # SEP + (chave, resposta) por query
        haystack_len = seq_len - query_len
        if haystack_len < 2 * n_keys:
            raise ValueError("seq_len pequeno demais para as agulhas + queries.")

        # palheiro = ruído variado (cada posição um distrator sorteado).
        seq = [rng.choice(filler_vocab) for _ in range(haystack_len)]
        # posições aleatórias para as agulhas no palheiro. Sorteamos slots em
        # posições PARES (slot=2j) para que (chave em 2j, valor em 2j+1) nunca
        # colida com a agulha vizinha — antes, slots adjacentes sobrescreviam o
        # valor da agulha anterior (~6% dos exemplos), criando um teto
        # silencioso de acurácia (auditoria 2026-06-12).
        slots = sorted(2 * j for j in rng.sample(range(haystack_len // 2), n_keys))
        for slot, k in zip(slots, keys):
            seq[slot] = ko + k
            seq[slot + 1] = vo + kv[k]
        label = [-1] * haystack_len

        # bloco de query
        seq.append(SEP); label.append(-1)
        for qk in q_keys:
            ans = vo + kv[qk]
            seq.append(ko + qk); label.append(ans)   # query-chave prevê resposta
            seq.append(ans); label.append(-1)

        seq, label = seq[:seq_len], label[:seq_len]
        if len(seq) < seq_len:
            pad = seq_len - len(seq)
            seq += [PAD] * pad; label += [-1] * pad
        X.append(seq); L.append(label)

    return (torch.tensor(X, dtype=torch.long, device=device),
            torch.tensor(L, dtype=torch.long, device=device), total)


def generate_var_tracking(n_examples=200, seq_len=512, n_hops=4,
                          vocab_size=256, device="cpu", seed=42):
    """
    Variable Tracking: define X0=val; X1=X0; ...; X_{n}=X_{n-1}; pergunta X_n.
    Tokens: usamos chaves como nomes de variáveis e valores como conteúdos.
    Sequência: (Xname, value/refname) pares + SEP + query Xn -> valor final.
    """
    rng = random.Random(seed)
    ko, vo, total = _layout(vocab_size)
    REF = total  # token marcador "isto é uma referência a outra variável"
    total += 1

    X, L = [], []
    for _ in range(n_examples):
        names = rng.sample(range(vocab_size), n_hops + 1)
        init_val = rng.randint(0, vocab_size - 1)
        seq, label = [], []
        # X0 = init_val
        seq += [ko + names[0], vo + init_val]; label += [-1, -1]
        # Xi = X_{i-1}  (representado por name do anterior, precedido de REF)
        for i in range(1, n_hops + 1):
            seq += [ko + names[i], REF, ko + names[i - 1]]
            label += [-1, -1, -1]
        # query: nome final -> valor inicial (que se propaga pela cadeia)
        seq.append(SEP); label.append(-1)
        ans = vo + init_val
        seq.append(ko + names[-1]); label.append(ans)
        seq.append(ans); label.append(-1)

        seq, label = seq[:seq_len], label[:seq_len]
        if len(seq) < seq_len:
            pad = seq_len - len(seq)
            seq += [PAD] * pad; label += [-1] * pad
        X.append(seq); L.append(label)

    return (torch.tensor(X, dtype=torch.long, device=device),
            torch.tensor(L, dtype=torch.long, device=device), total)


@torch.no_grad()
def evaluate_ruler(model, task="niah", batch_size=32, device="cpu", **gen_kw) -> dict:
    model.eval()
    gen = {"niah": generate_niah, "vt": generate_var_tracking}[task]
    inputs, labels, total_vocab = gen(device=device, **gen_kw)
    n = inputs.size(0)
    correct = tot = 0
    for i in range(0, n, batch_size):
        x, y = inputs[i:i + batch_size], labels[i:i + batch_size]
        logits, _ = model(x[:, :-1])
        target = y[:, :-1]
        mask = target != -1
        if mask.sum() == 0:
            continue
        preds = logits.argmax(dim=-1)
        correct += (preds[mask] == target[mask]).sum().item()
        tot += mask.sum().item()
    acc = correct / tot if tot else 0.0
    return {f"ruler_{task}_accuracy": acc, "correct": correct, "total": tot,
            "total_vocab": total_vocab, **gen_kw}


def selftest() -> bool:
    """Confere shapes, posições supervisionadas E corretude dos labels (oráculo).

    A checagem por oráculo é o que valida a mudança do palheiro (ruído variado):
    o oráculo do MQAR lê os pares (chave, valor) adjacentes e responde — só
    atinge 100% se os labels seguirem alinhados e o ruído NÃO colidir com as
    agulhas. Cobre o NIAH e o Variable Tracking.
    """
    from eval.mqar import _OracleModel
    ok = True

    xi, li, _ = generate_niah(n_examples=8, seq_len=256, n_keys=4, n_queries=2, vocab_size=64)
    sup = (li[:, :-1] != -1).sum().item()
    print(f"[niah] inputs {tuple(xi.shape)} supervisionadas={sup}")
    ok = ok and xi.shape == (8, 256) and sup == 8 * 2  # 2 queries por exemplo

    # Oráculo deve resolver o NIAH a 100% (labels corretos, ruído sem colisão).
    orac = _OracleModel()
    logits, _ = orac(xi[:, :-1])
    mask = li[:, :-1] != -1
    acc_niah = (logits.argmax(-1)[mask] == li[:, :-1][mask]).float().mean().item()
    print(f"[niah] oráculo acc={acc_niah:.3f} (esperado 1.000)")
    ok = ok and abs(acc_niah - 1.0) < 1e-9

    xv, lv, _ = generate_var_tracking(n_examples=8, seq_len=128, n_hops=3, vocab_size=64)
    supv = (lv[:, :-1] != -1).sum().item()
    print(f"[vt]   inputs {tuple(xv.shape)} supervisionadas={supv}")
    ok = ok and xv.shape == (8, 128) and supv == 8  # 1 query por exemplo

    print("  ✓ selftest OK" if ok else "  ✗ selftest FALHOU")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Subconjunto do RULER")
    parser.add_argument("--checkpoint")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--task", choices=["niah", "vt"], default="niah")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--n_examples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HybridModel(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(model.describe())
    gen_kw = {"n_examples": args.n_examples, "seq_len": args.seq_len}
    res = evaluate_ruler(model, task=args.task, batch_size=args.batch_size,
                         device=device, **gen_kw)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
