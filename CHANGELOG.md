# CHANGELOG — Reajuste e re-implementação do repositório

Insumo direto para a seção de Metodologia do TCC. Registra **o que mudou**,
**por quê** e **desvios** em relação ao plano original. Datas relativas
convertidas para absolutas (referência: 2026-06-02).

---

## Resumo

O repositório foi reajustado para (a) eliminar a recorrência de `loss: nan`,
(b) instalar os kernels Mamba-2 de forma robusta no Colab (ou cair para um
backend PyTorch puro defensável), (c) garantir paridade de parâmetros entre as
5 variantes e (d) tornar o treino resiliente a quedas de sessão. As decisões de
arquitetura fechadas (D1–D5) e o contrato público dos blocos foram preservados.

---

## config.py

- **Paridade por `d_ff` dual (decisão nova).** O bloco Mamba-2 carrega overhead
  fixo (in_proj/out_proj/conv) que o GQA não tem; com `d_model` compartilhado, o
  bloco Mamba fica mais pesado e as variantes com mais `M` estouram a banda de
  ±5%. Em vez de quebrar a exigência de `d_model` compartilhado, introduzimos
  **dois `d_ff`**: `d_ff` (atenção) e `d_ff_mamba` (menor), calibrados para
  igualar a contagem de parâmetros **por bloco**. Resultado analítico:
  `d_model=512, d_ff=2304, d_ff_mamba=1600` → ~50.1–50.3M ativos por variante,
  divergência máxima ~0.4% (bem dentro de ±5%). O valor final é confirmado por
  `check_parity.py` no Colab (com o backend real) e ajustável ali.
- **Campos Mamba-2 explícitos:** adicionados `headdim=64` e `chunk_size=256`
  (algoritmo SSD de Dao & Gu, 2024), além de validação em `__post_init__` que
  **falha alto** se `d_inner = expand*d_model` não for divisível por `headdim`.
- **Orçamento realista para Colab:** `max_tokens` reduzido de 15e9 (irreal em
  Colab Pro) para `1.5e9`. `block_size=1024` para o treino (a spec §5 sugere
  1024); `max_seq_len` do modelo mantido em 2048 para acomodar RULER/long-context
  na Fase C. `max_steps` é **derivado** de `max_tokens / tokens_por_step` quando
  não passado, garantindo o **mesmo orçamento de tokens** entre variantes.
- **Variantes renomeadas para os nomes da spec** (`attn_only`, `hybrid_3_1`,
  `hybrid_5_1`, `hybrid_7_1`, `ssm_only`) com **aliases** por razão (`0:12`,
  `3:1`, …) via `resolve_variant()`, mantendo compatibilidade com o fluxo antigo.
- **Correção dos padrões:** a tabela do CLAUDE.md continha caracteres cirílicos
  (`MMMА`); o **código** já usava ASCII e a contagem de `A` já batia com a razão.
  Conferido: 0:12→12A, 3:1→3A, 5:1→2A, 7:1→1A (~8%), 12:0→0A, sempre 12 blocos.

## setup_env.py (NOVO)

- Instala os kernels por **wheel pré-compilada** que casa exatamente com
  `(torch, cuda, cxx11abi, python)`, pulando a compilação. A flag `cxx11abi` é
  **derivada** de `torch._C._GLIBCXX_USE_CXX11_ABI` (causa #1 do erro
  `undefined symbol`), nunca chutada. Instala `causal-conv1d` antes de `mamba-ssm`.
- **Fallback explícito sem compilar:** se nenhuma wheel casar, define
  `MAMBA_BACKEND="torch"` e segue com o backend PyTorch puro. Loga claramente
  qual backend ficou ativo e expõe o resultado em `os.environ["MAMBA_BACKEND"]`.
- **Desvio em relação ao plano original:** abandonamos a rota `mamba2-minimal`
  + monkey-patch do `segsum` (que estava no notebook antigo) em favor do
  `transformers.Mamba2Mixer` — ver blocks.py.

## blocks.py

- **`MambaBlock` reescrito** com switch de backend via `MAMBA_BACKEND`:
  - `"kernels"` → `mamba_ssm.Mamba2` (fast path CUDA).
  - `"torch"` → `transformers.Mamba2Mixer` chamando `torch_forward`
    explicitamente (não depende da autodetecção do fast-path, que exigiria os
    kernels). **A arquitetura SSD subjacente é idêntica à de Dao & Gu (2024)**;
    muda apenas a implementação — defensável academicamente.
- Mapeamento `ModelConfig → Mamba2Config` explícito (`d_state`, `d_conv`,
  `expand`, `headdim`, `chunk_size`, `n_groups=1`) com **falha alta** na
  divisibilidade `d_inner % headdim`.
- O MLP do bloco Mamba usa `d_ff_mamba` (ver paridade). `RMSNorm`, `MLP` e
  `GQABlock` **não foram alterados** além do necessário; o contrato
  `(B,T,D)->(B,T,D)` foi preservado.

## model.py

- **Init escalado por profundidade (GPT-2):** as projeções que escrevem de volta
  no fluxo residual (`o_proj`, `down_proj`, `out_proj` do Mamba) são reescaladas
  por `1/sqrt(2 * n_layers)` após o init normal. Mantém a variância residual
  estável e é a principal medida contra `nan` em stacks profundos — relevante
  sobretudo para `ssm_only` e `hybrid_7_1`. Interface pública intacta.

## check_parity.py

- Passa a usar **alvo absoluto** de 50M ativos com banda ±5% e **falha
  (exit≠0)** se qualquer variante sair da banda, servindo de gate executável.
  Imprime tabela por variante e o spread relativo entre variantes.

## data/dataloader.py

- **Dataset trocado** de `cerebras/SlimPajama-627B` para **`DKYoon/SlimPajama-6B`**
  (D5), em **streaming** (evita baixar ~14 GB, sobrevive a reconexões).
- **Split de validação determinístico:** primeiras `VAL_NUM_SEQUENCES=256`
  sequências do split `validation`, `num_workers=0` (evita duplicação de stream
  em `IterableDataset`) → perplexidade comparável entre runs e variantes.
- Packing em blocos contíguos de `block_size`, EOT entre documentos, targets
  deslocados em 1.

## train.py

- Reescrito como **função importável** `train(variant_name, cfg) -> metrics`
  (sem `!python`; subprocesso não herda o backend no Colab). CLI fina por cima.
- **Resume automático** do último checkpoint (`last.pt`) com **escrita atômica**
  (`os.replace`) — assume que a sessão do Colab vai cair. Aponte `out_dir` para
  o Google Drive.
- **bf16 (autocast)** quando suportado; **fp32 + aviso** em T4. Nunca fp16 puro.
- AdamW (0.9, 0.95), weight decay 0.1 **sem decay** em params 1-D (normas/bias).
  Cosine com warmup por **frações** de `max_steps`. Grad clip global = 1.0.
- **Guarda contra `nan`:** o loop para com diagnóstico se a loss ficar não-finita,
  em vez de queimar GPU.
- Logging por intervalo: train loss, val loss/perplexidade, tokens/s, **pico de
  memória** e ms/step. Removido o `torch.compile()` incondicional (quebrava com o
  backend torch puro / em algumas GPUs do Colab).

## eval/ (NOVO pacote; migra benchmarks/ e evaluate.py)

- **`eval/mqar.py`** (prioridade alta): gerador sintético self-contained
  (`num_pairs`, `vocab`, `seq_len`) + **teste unitário** com modelo-oráculo que
  deve atingir 100% — valida labels e alinhamento.
  - **Correção de off-by-one (bug do código antigo):** os labels agora vivem no
    **mesmo espaço de índices** que os inputs (não deslocados); a avaliação
    compara `model(x[:, :-1])` contra `labels[:, :-1]`. Antes, o `labels[:, 1:]`
    desalinhava a posição supervisionada em uma casa (a resposta era cobrada do
    token SEP, não da query-chave).
- **`eval/perplexity.py`:** perplexidade no val fixo.
- **`eval/lambada.py`, `eval/hellaswag.py`:** loaders mínimos próprios
  (secundários). LAMBADA usa aproximação de último-token; para o número final de
  paper, preferir o lm-eval-harness.
- **`eval/ruler.py`** (último): subconjunto sintético — Multi-key NIAH +
  Variable Tracking, mesma convenção de labels do MQAR, com selftest.
- `benchmarks/mqar.py` e `evaluate.py` viraram **shims/dispatchers finos** sobre
  `eval/` para não quebrar imports e notebooks antigos.

## run_experiments.py (NOVO)

- Orquestra a ordem da §6 (Fase A gate → B → C), tudo inline/importável. A Fase A
  é um **gate**: smoke de blocos → paridade → smoke train (com teste de resume) →
  baselines. Só avança se tudo passar.

## run_colab.ipynb (NOVO)

- Notebook executável para o gate da Fase A no Colab: monta o Drive, instala deps,
  roda `setup_env`, smoke tests, paridade, selftests dos benchmarks e smoke train.

---

## Desvios e ressalvas (para a metodologia)

1. **Backend torch puro = `transformers.Mamba2Mixer`** (não `mamba2-minimal`).
   A SSD é a mesma de Dao & Gu (2024); o que muda é só a presença/ausência dos
   kernels CUDA. Com kernels o treino é ordens de magnitude mais rápido.
2. **Paridade verificada analiticamente** ao escrever o código (ambiente local
   Windows sem GPU/torch). A confirmação numérica final roda em
   `check_parity.py` no Colab; se a contagem real do `Mamba2Mixer` divergir,
   ajustar `d_ff_mamba` (o script falha alto e indica o ajuste).
3. **Orçamento de ~1.5B tokens** é um piloto viável em Colab Pro, não o regime de
   15B do plano original — escolhido para caber no prazo/GPU variável. O mesmo
   orçamento é aplicado a todas as variantes comparadas.
4. **LAMBADA** com aproximação de último-token (não última-palavra multi-token).

---

## Auditoria 2026-06-12 — confiabilidade, corretude e compatibilidade Colab

Revisão completa do repositório antes do gate da Fase A. Correções aplicadas:

### blocks.py — duas correções de corretude

- **[CRÍTICO] Init explícito de `A_log`/`dt_bias`/`D` no backend torch.**
  `Mamba2Mixer` instanciado diretamente (fora de um `Mamba2PreTrainedModel`)
  **não passa pelo `_init_weights` do HF**: em transformers 4.x o `dt_bias`
  ficava em 1.0 (Δt ≈ softplus(1) ≈ 1.31, fora da faixa [0.001, 0.1] do paper
  — risco de instabilidade); em transformers v5 os três params nascem como
  `torch.empty` (lixo de memória → nan imediato). `_Mamba2TorchMixer` agora
  replica o init oficial (dt ~ LogUniform[1e-3, 1e-1], A_log = log(1..H),
  D = 1), igual ao que o `mamba_ssm.Mamba2` faz no próprio `__init__` —
  **os dois backends ficam estatisticamente consistentes**.
- **RoPE na atenção (`use_rope=True` em ModelConfig, sem parâmetros — paridade
  D3 intacta).** O repositório não tinha NENHUM encoding posicional; Jamba e
  Nemotron-H treinam sem posicional explícito porque os blocos Mamba dão ordem,
  mas a baseline `attn_only` pura ficaria NoPE — enfraquecida, enviesando a
  comparação a favor das variantes com SSM (confundidor da variável proporção).
  `use_rope=False` reproduz o regime NoPE estilo Jamba se desejado.
- **Atenção via `F.scaled_dot_product_attention` (is_causal=True).** A
  implementação anterior materializava a matriz T×T por cabeça (em fp32/T4,
  ~0.5 GB por camada no forward em B=16, T=1024 → OOM provável). SDPA usa
  Flash/mem-efficient attention; o buffer `causal_mask` (4 MB por bloco no
  state_dict) foi removido. Checkpoints antigos (não havia nenhum de valor)
  ficam incompatíveis pela chave do buffer.

### data/dataloader.py — correção de duplicação + robustez

- **[BUG] Sharding por worker no stream.** `IterableDataset` com
  `num_workers=2` fazia cada worker iterar o stream INTEIRO → cada sequência
  aparecia 2× no treino (diversidade efetiva pela metade, gradientes
  correlacionados). Agora os documentos são particionados por
  `i % num_workers == worker_id` (orçamento de tokens dividido entre shards).
- **Held-out de validação cacheado em memória** (`get_val_batches`): ~2 MB;
  evita reabrir o stream HF a cada `eval_interval` (lento e uma falha de rede
  no meio derrubaria a run). `train.py` e `eval/perplexity.py` usam o cache.

### eval/ruler.py — correção do gerador NIAH

- **[BUG] Colisão de agulhas adjacentes.** Slots eram sorteados livremente;
  quando dois slots eram consecutivos, a chave da agulha seguinte sobrescrevia
  o valor da anterior (~6% dos exemplos com os defaults) — teto silencioso de
  acurácia. Slots agora são sorteados em posições pares (chave em 2j, valor em
  2j+1), sem colisão possível.

### setup_env.py / requirements.txt — compatibilidade Colab (verificada)

- **Ambiente Colab atual (jun/2026): Python 3.12, PyTorch 2.8, CUDA 12.x.**
  Verificado que os releases `mamba-ssm v2.3.2.post1` (mai/2026) e
  `causal-conv1d v1.6.2.post1` publicam wheels para
  **cu12 × torch2.8 × cxx11abiTRUE × cp312** (e torch 2.6–2.10, cp310–313,
  cu11/cu12/cu13) — ou seja, **o backend kernels voltou a ser viável no
  Colab**, ao contrário do que o plano.md (Fase 3, Problema 1) registrou para
  o início do projeto. Constantes de versão atualizadas de 2.3.0/1.6.0 para
  2.3.2.post1/1.6.2.post1. O fallback torch puro permanece como rede de
  segurança automática.
- `datasets` repinado de `<3` para `>=3.0,<6` (o pin antigo forçaria downgrade
  de fsspec/huggingface_hub no Colab; o dataset é parquet puro e funciona em
  3.x/4.x). `transformers>=4.44,<5` mantido (API 4.x validada; v5 coberto pelo
  init explícito de qualquer forma).

### Ressalvas registradas (sem mudança de código)

- **MQAR/RULER são avaliações zero-shot** sobre um LM treinado em texto: o
  mecanismo esperado é cópia por induction heads (o par consultado está
  literalmente no contexto). As acurácias absolutas serão baixas; o que
  interessa é o CONTRASTE entre variantes. Registrar na metodologia.
- **Resume não retoma a posição do stream de dados** (recomeça do início do
  SlimPajama): após uma queda, parte dos tokens é revisitada. Aceitável para o
  orçamento piloto; registrar como limitação.
- **Throughput**: medir tok/s no smoke train e recalibrar `max_tokens` se o
  backend ativo for o torch puro (SSD sem kernels é várias vezes mais lento;
  1.5B tokens pode não caber no prazo — com kernels, viável).

---

## Fase A no Colab (2026-06-12) — resultados do gate e correções pós-run

### Resultados (A100-SXM4-40GB, bf16, backend torch puro)

- **Gate A2–A4 passou inteiro**: smoke de blocos sem nan, paridade
  50.14–50.34M (spread 0.4%), selftests MQAR (oráculo 100%) e RULER OK,
  dataloader OK, smoke train com queda de loss e resume funcionando.
- **`attn_only` COMPLETOU o orçamento de 1.5B tokens**: 11 444 steps a
  ~116k tok/s (~3h35 de A100), val_loss 3.42, **perplexidade 30.61**, sem nan.
  Checkpoint em `hybrid_ckpts/attn_only/last.pt` — NÃO retreinar.
- **Kernels: 404 na wheel.** O runtime do Colab subiu para **torch
  2.11.0+cu12.8** e os releases (mai/2026) publicam wheels só até `torch2.10`.
  O fallback torch puro assumiu corretamente.
- **`ssm_only` e `hybrid_3_1`: OOM no 1º step.** O `torch_forward` do
  `Mamba2Mixer` materializa `(M[..., None] * hidden_states[:, :, None])`
  com shape (B, n_chunks, chunk, chunk, heads, head_dim) em fp32 — com B=16 e
  T=1024 é UMA alocação de ~16 GiB; não cabe nem em A100 40GB.

### Correções aplicadas

- **setup_env.py — fallback de minor do torch nas wheels.** Quando o torch do
  runtime é mais novo que a última wheel publicada, tenta as wheels dos minors
  anteriores (2.10 → 2.9 → … → 2.6, nunca um minor mais novo); o smoke test
  (import + forward em CUDA) decide se a ABI casa. Se nenhum casar, cai para o
  backend torch como antes.
- **train.py / config.py — micro-batch adaptativo no backend torch.** Com
  `MAMBA_BACKEND=torch` e blocos `M` no padrão, `batch_size` é reduzido por
  halving até `mamba_torch_microbatch` (default 4) e `grad_accumulation_steps`
  é multiplicado na mesma proporção — **tokens/step e batch efetivo idênticos**
  aos das demais variantes (comparabilidade preservada; muda só o quanto cabe
  na GPU por passada). `torch.cuda.empty_cache()` no início de `train()` limpa
  resíduos de runs anteriores na mesma sessão (o OOM da Fase B herdou memória
  da tentativa falha do ssm_only).

### Observação de viabilidade

Com micro-batch 4 o backend torch deve caber com folga na A100, mas o
throughput do SSD puro PyTorch é incógnita — **medir tok/s nos primeiros ~100
steps do ssm_only**: abaixo de ~30k tok/s, o orçamento de 1.5B tokens custa
>14h de A100 por variante Mamba (×4 variantes) e convém (a) priorizar o
caminho de kernels via fallback de minor, (b) reduzir `max_tokens` para todas
as variantes (attn_only incluída, retreinada no orçamento menor), ou (c)
treinar fora do Colab. Decisão a registrar quando houver o número medido.

---

## Fase A+B no Colab (2026-06-15) — run completa com backend KERNELS

### Resultados (NVIDIA L4, bf16, backend **kernels**)

O fallback de minor do torch nas wheels (registrado acima) **funcionou**: torch
2.11 deu 404, o setup tentou as wheels de torch 2.10, o smoke forward passou e o
backend ativo foi **kernels** (não torch puro). Sem OOM. As três variantes do
núcleo treinaram o orçamento de 1.5B tokens (11 444 steps):

| Variante   | val ppl | MQAR acc (ponto único) | throughput |
|------------|---------|------------------------|------------|
| attn_only  | 30.61   | 0.393                  | —          |
| hybrid_3_1 | 29.07   | 0.372                  | ~30k tok/s |
| ssm_only   | 32.60   | 0.277                  | ~110k tok/s|

Faltam `hybrid_5_1` e `hybrid_7_1` (Fase C). Paridade 50.14–50.34M, sem nan.

### Leitura crítica (insumo da discussão do TCC)

- **PPL na ordem esperada** (Jamba): híbrido < atenção pura < SSM puro. O ganho
  de PPL do `hybrid_3_1` sobre o `attn_only` é o achado mais sólido.
- **O gap de recall SSM↔atenção é PEQUENO** (0.393 vs 0.277) — não o colapso que
  Zoology (Arora et al. 2023) prevê — e o `hybrid_3_1` NÃO supera o `attn_only`
  no MQAR. Suspeita forte de **artefato de medição**, não resultado: o MQAR foi
  avaliado num ÚNICO ponto (seq_len=128, n_pairs=8) — justamente o regime em que
  as arquiteturas mais se parecem — e sem nível de acaso reportado. Em Zoology o
  gap só *abre* ao varrer seq_len/n_pairs.

## Avaliação 2026-06-16 — varredura de recall (eval/recall_sweep.py)

- **NOVO `eval/recall_sweep.py`** (só inferência; não retreina): varre o MQAR num
  GRID (seq_len × n_pairs) e o NIAH do RULER em vários seq_len, reportando o
  **nível de acaso** (1/vocab_size) e contagens por célula. Reaproveita os
  geradores já validados (`generate_mqar_examples`, `evaluate_ruler`) **sem
  alterar a interface pública**; a semântica de avaliação (argmax no vocab
  inteiro, `model(x[:,:-1])` vs `labels[:,:-1]`) é idêntica. Inclui `--selftest`
  com o modelo-oráculo (1.0 em toda célula viável; pula células onde o prefixo
  não cabe). **Por quê:** decidir se o gap pequeno é treino subótimo ou medição
  no regime errado exige a CURVA de Zoology, não um ponto. É o teste que
  distingue "não há efeito" de "medi no lugar errado" — pré-requisito para
  qualquer mudança no treino.
- **NOVO `run_experiments.phase_b_recall(out_root)`**: runner inline (estilo
  `phase_b`, sem `!python`) que roda o sweep sobre os checkpoints existentes no
  Drive e salva um JSON por variante; pula os ausentes (5_1/7_1 antes da Fase C).
  CLI: `--phase b_recall`.
- **Decisão de sequência (metodológica):** NÃO se altera o código de TREINO antes
  de rodar este sweep. Se o gap atenção↔SSM abrir com seq_len/n_pairs, a medição
  estava certa e o treino não muda (segue Fase C). Se permanecer chato no grid
  inteiro, aí há evidência para investigar treino/dados.

### Resultado do sweep (Colab, L4, kernels) — hipótese central CONFIRMADA

O grid resolveu o diagnóstico: **o gap de recall abre monotonicamente com
`n_pairs`** (carga de memória associativa), exatamente como Zoology (Arora 2023)
prevê. Em seq_len=1024 (acaso=0.002):

| n_pairs | attn_only | hybrid_3_1 | ssm_only |
|---------|-----------|------------|----------|
| 4       | 0.349     | 0.380      | 0.309    |
| 8       | 0.392     | 0.367      | 0.282    |
| 16      | 0.361     | 0.335      | 0.192    |
| 32      | 0.288     | 0.249      | 0.083    |
| 64      | 0.217     | 0.181      | 0.024    |

O `ssm_only` despenca para perto do acaso (estado recorrente de tamanho fixo
satura); a atenção degrada com graça. O ponto único da phase_b (n_pairs=8)
caía no regime em que as arquiteturas mais se parecem — por isso o gap parecia
pequeno. **`seq_len` é inerte** no MQAR atual (pares empacotados contíguos no
início + PAD; a distância não varia, só a quantidade) — a variável de estresse
efetiva é `n_pairs`. Registrar como limitação do gerador MQAR.

### NIAH — artefato encontrado e corrigido (eval/ruler.py)

- **[BUG de avaliação] `attn_only` zerava no NIAH (0/800)** enquanto Mamba/híbrido
  pontuavam — paradoxal (atenção global deveria liderar NIAH). Causa: o palheiro
  era `[FILLER] * haystack_len`, ou seja **um ÚNICO token repetido por ~80% da
  sequência**. Nenhum LM treinado em texto natural vê isso, e é ADVERSÁRIO à
  atenção: uma sequência constante colapsa o padrão causal (todas as chaves/
  valores iguais → a query não destaca a agulha). O Mamba é robusto (seleção via
  Δt ignora o ruído). O RULER real (NVIDIA, Hsieh et al. 2024) usa ruído VARIADO
  (frases de ruído / ensaios), nunca um token único.
- **Correção:** palheiro agora amostra de um vocabulário de `n_filler=64`
  tokens-distrator DISJUNTO de chaves/valores (sem colisão com agulhas). Validado:
  oráculo do MQAR resolve o NIAH corrigido a 100% (labels intactos); o selftest
  do ruler.py passou a checar isto por oráculo (antes só conferia shapes).
- **Impacto:** o resultado NIAH da run de 15/jun (attn=0.000) era INVÁLIDO (artefato).
  O MQAR daquela run permanece VÁLIDO (gerador não tocado) e é o resultado principal.

### Re-run do sweep com NIAH corrigido (2026-06-16) — paradoxo resolvido

Re-rodado sobre os MESMOS checkpoints (só o gerador NIAH mudou). O `attn_only`
saiu de 0.000 para a faixa de topo, ordenação agora coerente com a literatura
(atenção > híbrido ≫ SSM em recall longo):

| NIAH seq_len | attn_only | hybrid_3_1 | ssm_only | acaso |
|--------------|-----------|------------|----------|-------|
| 256          | 0.236     | 0.203      | 0.023    | 0.002 |
| 512          | 0.184     | 0.125      | 0.010    | 0.002 |
| 1024         | 0.225     | 0.126      | 0.006    | 0.002 |

O `ssm_only` cola no acaso (estado fixo não retém agulhas espalhadas). MQAR
inalterado (confirma a estabilidade do gerador). **Quadro final do TCC:** em
recall puro (MQAR/NIAH) a atenção lidera e o SSM colapsa sob carga (Zoology,
Arora 2023); em PPL de texto natural o híbrido vence ambos os extremos —
29.07 < 30.61 (attn) < 32.60 (ssm) — (Jamba). Trade-off memória↔atenção
demonstrado com paridade ~50M (±0.4%) e 1.5B tokens. Falta a Fase C (5_1/7_1)
para a curva completa da proporção.

## Melhorias de avaliação 2026-06-17 — distância (MQAR gap_fill) + PPL no JSON

Duas melhorias aplicadas (só avaliação; treino intocado). Validadas localmente
por oráculo (acc=1.0 nos dois modos e em todos os seq_len) e end-to-end.

### 1. `seq_len` efetivo no MQAR — modo `gap_fill` (eval/mqar.py)

- **Limitação corrigida:** no MQAR clássico os pares ficavam contíguos no início
  e o resto era PAD, então `seq_len` NÃO alterava a distância chave→query — a
  única variável de estresse era `n_pairs` (CARGA). Isso explica por que o eixo
  `seq_len` saía inerte no primeiro sweep.
- **`generate_mqar_examples(..., gap_fill=False, n_filler=64)`** (parâmetro novo,
  default retrocompatível). Com `gap_fill=True`, tokens-distrator variados (espaço
  DISJUNTO de chaves/valores) são intercalados ENTRE os pares, empurrando as
  chaves para longe das queries → a distância escala com `seq_len` (estilo RULER).
  Chave e valor permanecem ADJACENTES (o ruído vai entre pares), preservando os
  labels. Verificado: SEP migra de 47→239→1007 para seq_len 64→256→1024.
- **`_OracleModel`** passou a parear por janela deslizante de passo 1
  (`kv[t]=t+1`), robusta ao ruído (pares espúrios entram no dict mas nunca são
  consultados; chave→valor sempre registrado). Resolve os DOIS modos a 100%.
- **Interpretação:** agora há DOIS eixos ortogonais — `n_pairs` (carga, modo pack)
  e `seq_len` (distância, modo gap). Zoology mostra o gap SSM↔atenção abrindo em
  ambos; o pack já confirmou a carga, o gap mede a distância.

### 2. Sweep consolidado + perplexidade (eval/recall_sweep.py)

- **`sweep_checkpoint` agora roda os DOIS grids** (`mqar_grid_pack` +
  `mqar_grid_gap`) e inclui a **perplexidade** no val fixo (`with_perplexity=True`,
  reusa `eval/perplexity.py`). Um único JSON por variante reúne PPL + MQAR(carga)
  + MQAR(distância) + NIAH — Resultados do TCC saem de um arquivo só.
- **`phase_b_recall`** atualizado: resumo por variante imprime PPL, célula MQAR-pack
  mais difícil e NIAH mais longo. JSON renomeia `mqar_grid`→`mqar_grid_pack`/`_gap`.
- Selftests de `eval/mqar.py`, `eval/ruler.py` e `eval/recall_sweep.py` cobrem
  ambos os modos por oráculo; todos passam.

---

## Estado do repositório, resultados e pendências (síntese 2026-06-17)

### O que foi feito no repositório (avaliação; treino intocado)
- `eval/recall_sweep.py` (NOVO): grid MQAR em dois modos (carga/distância), sweep
  NIAH, perplexidade e nível de acaso; JSON consolidado por variante; `--selftest`.
- `eval/mqar.py`: modo `gap_fill` (distância ~ seq_len); oráculo robusto a ruído.
- `eval/ruler.py`: palheiro NIAH agora é ruído variado (corrige o artefato do
  FILLER único que zerava o `attn_only`); selftest valida labels por oráculo.
- `run_experiments.py`: `phase_b_recall()` + CLI `--phase b_recall`.
- `train.py`/`config.py`/`model.py`/`blocks.py`: **NÃO alterados** — nenhum
  problema de treino foi identificado; os dois artefatos eram de avaliação.

### Resultados registrados (run 2026-06-15, L4, kernels, 1.5B tokens, ~50M ±0.4%)
- **Perplexidade (texto):** hybrid_3_1 **29.07** < attn_only 30.61 < ssm_only 32.60.
- **MQAR (carga, n_pairs↑, seq_len=1024):** attn ≥ hybrid ≫ ssm; ssm cai a ~acaso
  (0.024 em n_pairs=64) — colapso de Zoology confirmado.
- **NIAH corrigido (distância):** attn 0.225 > hybrid 0.126 ≫ ssm 0.006 (@1024).
- **Leitura:** recall puro → atenção lidera; LM de texto → híbrido vence ambos.
- JSONs em `MyDrive/hybrid_ckpts/_recall_results/`.

### O que falta
1. ~~Re-rodar `phase_b_recall` no formato novo~~ — FEITO (2026-06-17). Os 3 JSONs
   (PPL + mqar_grid_pack + mqar_grid_gap + niah_sweep) estão em
   `MyDrive/hybrid_ckpts/_recall_results/`, verificados no Drive.
2. **Fase C — treinar `hybrid_5_1` e `hybrid_7_1`** (`R.phase_c`): completa o eixo
   da proporção (5 pontos) e cobre o sweet spot ~1:7 da literatura (Jamba/Nemotron-H).
3. **Sweep final** incluindo 5_1/7_1 (o runner os pega automaticamente).
4. Benchmarks secundários (lambada/hellaswag) e Variable Tracking do RULER, se
   houver orçamento — opcionais para o TCC.

### Resultado do eixo DISTÂNCIA (MQAR gap, 2026-06-17) — colapso total do SSM

O modo `gap` (distância chave→query ~ seq_len) revelou o resultado mais decisivo:
o SSM colapsa MAIS pela distância do que pela carga. Em n_pairs=8, variando a
distância (acaso=0.002):

| seq_len | attn_only | hybrid_3_1 | ssm_only |
|---------|-----------|------------|----------|
| 64      | 0.316     | 0.283      | 0.143    |
| 128     | 0.262     | 0.235      | 0.058    |
| 256     | 0.224     | 0.202      | 0.019    |
| 512     | 0.176     | 0.138      | 0.007    |
| 1024    | 0.242     | 0.116      | 0.001    |

O `ssm_only` chega ao ACASO (0.001) a ~1000 tokens de distância — o estado
recorrente de tamanho fixo "esquece" o par. A atenção é ~plana na distância
(acesso global); o híbrido degrada intermediário. **Os dois eixos de Zoology
(carga + distância) estão demonstrados.** Quadro consolidado das 3 variantes:
recall colapsa no SSM (pior na distância), atenção robusta, híbrido no meio;
PPL de texto o híbrido vence (29.07 < 30.61 < 32.60). Nota: `attn_only` tem
não-monotonia leve em seq_len=512→1024 (0.176→0.242), provável extrapolação de
RoPE perto do block_size=1024 de treino — registrar como nota de rodapé.

## data/dataloader.py — num_workers=0 por padrão (2026-06-17)

- **[BUG de ambiente] Treino do `hybrid_5_1` travou no Colab (A100): 15 min sem
  log nem uso de GPU, parado logo após "Iniciando treino...".** Diagnóstico
  isolou a causa: `load_dataset(streaming=True)` e o 1º documento respondem em
  ~2s em processo único, mas `make_dataloader` usava `num_workers=2` →
  multiprocessing, e cada worker reabre o stream HF no 1º `next()`. No Colab
  esses subprocessos travam silenciosamente (erros/prints suprimidos) e o 1º
  batch nunca chega — daí GPU em 0% e nenhum log (o 1º log só sai no step 20).
- **Correção:** `make_dataloader(..., num_workers=0)` por padrão — o stream roda
  no processo principal (confiável). O gargalo real é a GPU, não a CPU/IO (o
  tok/s das runs anteriores não era CPU-bound), então o impacto no throughput é
  mínimo. `num_workers>0` continua disponível (reativa o sharding por documento).
- **Nota de comparabilidade (metodologia):** com `num_workers=2` o
  `_stream_tokens` repartia `max_tokens` entre 2 shards e cada worker via
  documentos alternados (`i % 2`); com `num_workers=0` o processo vê TODOS os
  documentos até `max_tokens`. **Tokens treinados idênticos (1.5B); o conjunto de
  documentos amostrados difere** entre as 3 variantes já treinadas (workers=2) e
  as novas 5_1/7_1 (workers=0). Efeito desprezível — 1.5B é uma fração mínima do
  SlimPajama-6B i.i.d.; ambas são amostras aleatórias representativas. Registrado
  por transparência; não justifica retreinar as 3 anteriores.

## Fase C concluída (2026-06-21) — 5 variantes, curva completa da proporção

`hybrid_5_1` e `hybrid_7_1` treinados (1.5B tokens, kernels) e sweep das 5
variantes feito. Resultado final do experimento.

### Perplexidade (val fixo) — ótimo interior na proporção

| variante   | M:A  | PPL    |
|------------|------|--------|
| attn_only  | 0:12 | 30.61  |
| hybrid_3_1 | 9:3  | 29.07  |  ← melhor
| hybrid_5_1 | 10:2 | 29.64  |
| hybrid_7_1 | 11:1 | 29.52  |
| ssm_only   | 12:0 | 32.60  |

Os 3 híbridos (29.07–29.64) batem ambos os extremos puros; platô estreito entre
eles (leve não-monotonia 5_1>7_1, dentro do ruído). Reproduz o achado de Jamba
("1:3 a 1:7 = sweet spot, sem diferença substancial") em escala ~50M/1.5B tokens.

### Recall (acc@1; acaso=0.002)

| variante   | MQAR pack n=64 | MQAR gap n=8 sl=1024 | NIAH sl=1024 |
|------------|----------------|----------------------|--------------|
| attn_only  | 0.217          | 0.242                | 0.225        |
| hybrid_3_1 | 0.181          | 0.116                | 0.126        |
| hybrid_5_1 | 0.252          | 0.265                | 0.244        |
| hybrid_7_1 | 0.233          | 0.238                | 0.243        |
| ssm_only   | 0.024          | 0.001                | 0.006        |

**Achado inesperado (registrar e investigar):** `hybrid_5_1` e `hybrid_7_1`
IGUALAM ou SUPERAM o `attn_only` em recall (pack/gap/NIAH), e o `hybrid_3_1`
fica ATRÁS dos outros dois híbridos — uma INVERSÃO da expectativa ingênua de que
"mais atenção = mais recall". Hipótese principal: as posições das 2-3 camadas de
atenção no stack importam mais que a quantidade. No 3_1 (MMMAMMMAMMMA) a 1ª
atenção só aparece na 4ª camada; no 5_1/7_1 os blocos Mamba pré-atenção podem
estar "resumindo" melhor o contexto antes da camada de atenção fazer o lookup
(efeito de roteamento Mamba→atenção, reportado em análises do Jamba/Nemotron-H).
NÃO é artefato de medição (3 tarefas independentes concordam; ssm_only colapsa e
attn_only é robusto, como esperado). É um resultado científico próprio — merece
uma figura (acc × proporção) e discussão; vale checar com seeds adicionais do
gerador MQAR para barras de erro antes de afirmar a inversão como definitiva.

### Quadro consolidado do TCC
- **PPL (texto):** híbrido vence os extremos; ótimo em ~3:1, platô até 7:1 (Jamba).
- **Recall (MQAR/NIAH):** ssm_only colapsa (gap→acaso); atenção robusta; híbridos
  5_1/7_1 no topo. O gap por DISTÂNCIA é o eixo mais discriminante (ssm 0.001).
- Trade-off memória↔atenção demonstrado nas 5 proporções, paridade ±0.4%, mesmo
  orçamento. **Pendências:** barras de erro no MQAR (multi-seed); benchmarks
  secundários (lambada/hellaswag) opcionais.

## Avaliação multi-seed 2026-06-22 — barras de erro (eval/recall_sweep.py)

Os números de recall acima vinham de UMA seed do gerador por célula — sem desvio
não dá para afirmar a "inversão" (5_1/7_1 > attn; 3_1 atrás) contra a variância
de amostragem. Implementado multi-seed (só avaliação; treino intocado):

- **`mqar_grid` / `niah_sweep` / `sweep_checkpoint` agora aceitam `seeds`**
  (default `(0,1,2,3,4)`). Cada célula é reavaliada com N conjuntos de exemplos
  (seeds distintas) e reporta **`accuracy` (média), `acc_std` (dp amostral) e
  `acc_per_seed`** no JSON. `n_seeds=1` reproduz o ponto único anterior. O seed
  entra nos geradores já existentes (`generate_mqar_examples(seed=)`,
  `evaluate_ruler(..., seed=)` → `generate_niah`), sem alterar interface pública.
- **Validação:** selftest do oráculo passou nos dois modos (média 1.0, dp 0.0,
  `acc_per_seed` todos 1.0); end-to-end com modelo real produz o JSON com os
  campos novos. CLI ganhou `--n_seeds`.
- **Custo:** ~5× o tempo de inferência do sweep (≈45–60 min de GPU p/ as 5
  variantes com 5 seeds) — aceitável, só inferência.
- **Próximo:** re-rodar `phase_b_recall` para gravar os JSONs COM barras de erro;
  só então afirmar/derrubar a inversão 5_1/7_1>attn e a ordenação fina entre
  variantes de topo. Diferenças vs ssm_only já são enormes (não dependem disto).

## Resultados finais com barras de erro (2026-06-23) — 5 seeds, sweep completo

Sweep multi-seed re-rodado nas 5 variantes (5 seeds; MQAR n=512/seed, NIAH
n=200/seed). Desvios pequenos (≤0.015); significância por Welch t (gl~4,
|t|>2.78 ⇒ p<0.05). Médias ± dp:

| variante   | PPL   | MQAR carga (n=64) | MQAR dist. (sl=1024) | NIAH (sl=1024) |
|------------|-------|-------------------|----------------------|----------------|
| attn_only  | 30.61 | 0.2205 ± 0.0023   | 0.1148 ± 0.0081      | 0.2310 ± 0.0089|
| hybrid_3_1 | 29.07 | 0.1832 ± 0.0016   | 0.2312 ± 0.0027      | 0.1157 ± 0.0150|
| hybrid_5_1 | 29.64 | 0.2525 ± 0.0029   | 0.2652 ± 0.0124      | 0.2570 ± 0.0151|
| hybrid_7_1 | 29.52 | 0.2358 ± 0.0019   | 0.2628 ± 0.0072      | 0.2297 ± 0.0116|
| ssm_only   | 32.60 | 0.0242 ± 0.0008   | 0.0023 ± 0.0008      | 0.0035 ± 0.0027|

**Conclusões estatisticamente sustentadas (a "inversão" é REAL):**
- **hybrid_5_1/7_1 SUPERAM o attn_only em recall** — MQAR-carga: 5_1 vs attn
  diff +0.032 (t≈19); NIAH: 5_1 vs attn +0.026 (t≈3.3). Não é ruído.
- **hybrid_5_1 > hybrid_7_1** no MQAR-carga (t≈11) e NIAH (t≈3.2); empatam na
  distância (t≈0.4, n.s.). O sweet spot fino de recall é o **5_1 (~17% atenção)**.
- **hybrid_3_1**: pior que attn em carga/NIAH (t≈-15 a -30), porém MELHOR na
  distância (gap 0.231 vs 0.115, t≈-30) — a *distribuição* das camadas de
  atenção, não só a quantidade, modula o tipo de recall.
- **ssm_only colapsa** nos três eixos (distância → acaso 0.002). Zoology
  confirmado sem ambiguidade.

**Quadro do TCC:** (1) SSM puro tem memória estruturalmente limitada; (2) a
atenção pura NÃO é o teto de recall — híbridos a superam (roteamento Mamba→
atenção, cf. Nemotron-H/Jamba); (3) há ótimo INTERIOR na proporção — PPL em ~3:1,
recall em ~5:1, nunca os extremos. Paridade ±0.4%, 1.5B tokens, 5 seeds.

- **NOVO `plot_results.py`** (matplotlib): lê os 5 JSONs de `_recall_results/` e
  gera a figura `acc × proporção` com barras de erro (4 painéis: PPL, MQAR-carga,
  MQAR-distância, NIAH). Para rodar no Colab ou local. Não altera nada do experimento.
