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
