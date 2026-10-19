# bnk

A from-scratch inference engine for **Qwen3.8-Flash-Next-class mixture-of-experts models** (`qwen4exp`) on a
single **NVIDIA V100** with lots of host RAM. The model does not fit in VRAM, so bnk splits it across tiers: the
dense weights and the hottest experts live on the GPU, every expert lives in pinned system RAM, misses are computed
by the CPU during decoding, and experts stream over PCIe while a prompt is read.

With **two (or more) V100s** the model's layers are split between them (a pipeline): two 32 GB cards hold every
expert, so the CPU leaves the decode path and prompts are read on both cards at once (`docs/DUAL-GPU.md`).

It also runs the **dense Qwen3.8 models** (`qwen35`: Qwen3.8-27B) through a second engine that keeps every weight
in VRAM, tensor parallel over the two cards (`docs/DENSE.md`): Q8_0 at 104 tok/s with its own MTP drafter.

On top of that: speculative decoding with the model's own MTP layer, a 256K context whose memory is lent to the
expert cache until it is needed, **several agents served at once** (their conversations decoded together in one
batch, and parked in host RAM between turns so a turn reads only its new tokens), a thinking-loop guard, an
OpenAI- and Anthropic-compatible server, and a live dashboard.

It is built for, and measured on, this machine:

| Component | Specification |
|---|---|
| GPU | Tesla V100 32 GB (PG500-216, sm_70), HBM 889 GB/s measured |
| CPU | AMD EPYC 7402, 32 vCPUs (AVX2, no AVX-512) |
| RAM | 128 GB DDR4, ~99 GB/s measured at 24 threads |
| Link | PCIe 3.0 x16, 13.1 GB/s host-to-device measured |
| OS / CUDA | Ubuntu 24.04, driver 580, CUDA 12.8 (CUDA 13 dropped sm_70) |

**Contents:** [Quick start](#quick-start) · [Performance](#performance) · [How it works](#how-it-works) ·
[Models](#models) · [Setup in detail](#setup-in-detail) · [Using it](#using-it) ·
[Troubleshooting](#troubleshooting) · [Repository](#repository) · [Limitations](#limitations)

## Quick start

On a fresh Ubuntu 22.04 or 24.04 machine with a V100 and at least 64 GB of RAM:

```bash
git clone https://github.com/christ-pher/bnk.git
cd bnk
./setup.sh --mtp
```

`setup.sh` asks before it changes anything. On a machine without the NVIDIA driver it offers to install the
driver and CUDA 12.8, then tells you to reboot; **reboot and run `./setup.sh --mtp` again** to finish. At the end
it offers to download the default model (84 GB). Then:

```bash
./run.sh            # pick a model from the list (or: ./run.sh iq3_s)
```

and open `http://<this machine's IP>:8080` for the dashboard. The OpenAI-compatible API is at `/v1`, the
Anthropic one at the root. The first start takes about a minute. Details, options and problems:
[Setup in detail](#setup-in-detail) and [Troubleshooting](#troubleshooting).

## Performance

All numbers were measured on the machine above. "Greedy" runs decode deterministically with a warm expert cache;
"served" runs go through the HTTP server with each model's default sampling (`configs/`), which is what a chat
client sees.

### Qwen3.8-27B (dense) on the two V100s

`unsloth/Qwen3.8-27B-GGUF` Q8_0 (27 GiB, every weight in VRAM) through the dense engine (`docs/DENSE.md`): the
model's layers are the Flash-Next blocks without the experts, so the same kernels run it. Its layer outputs match
llama.cpp's fp32 path within ~1% (the int8 activation quantization), same top token everywhere tested.

| Greedy, Q8_0 | Layer pipeline (`--gpus 0,1`) | Tensor parallel (`--tp 1`, the config's default) |
|---|---|---|
| Decode without drafting | 30.8 tok/s | **42.4 tok/s** |
| Decode with the model's MTP block drafting | 80.6 tok/s (3.4 tokens per round) | **104 tok/s** |
| 3,000-token prompt | 1,100 tok/s | **1,330 tok/s** |
| 12,000-token prompt | | **1,309 tok/s** (869 with the CUDA-core attention kernel) |
| 64K / 250K-token cold read | | 970 / 531 tok/s |
| Decode with drafting at 64K / 250K of context | | 85 / 54 tok/s |
| Served (sampled, thinking, 545-token answer) | | 87.5 tok/s |

Decode is weight-bandwidth-bound (each card streams its 13 GB per forward); tensor parallelism lets both cards
stream at once, with 128 host-memory exchanges of the partial sums per forward (23 us each) in between.

### What lookahead pipelining changed (2026-10-19)

On the two-GPU pipeline a single stream used each card half the time: GPU 0's layers, then GPU 1's, and the next
window waits for the verdict. Lookahead runs the next window on GPU 0 while GPU 1 verifies this one, assuming
every draft is accepted, against a shadow copy of the recurrent state; when the verdict agrees (the drafter's
confidence, or a lookup match, decides when to try) the round shrinks to GPU 1's half, and when it does not,
nothing was touched. Copy-heavy output gains a fifth (a verbatim file rewrite on FP4-Local: 177 -> 209 tok/s,
86% of speculations confirmed), code a few percent, prose nothing. `docs/LOOKAHEAD.md`; `BNK_LOOKAHEAD=0`
turns it off.

### What prompt-lookup drafting and prefix forking changed (2026-10-19)

Two of the opportunities in `docs/PERF-IDEAS-2026-10-19.md`, details and measurements in `docs/LOOKUP-FORK.md`:

* **Prompt-lookup drafting.** When the last three tokens occurred earlier in the conversation, what followed
  them is proposed as up to 7 drafts for the verify window (the MTP drafter proposes 4, one step each); the
  MTP's first draft arbitrates when the match is short. Copy-heavy output, what coding agents produce when they
  re-emit files, goes from 4.6-4.9 to 7.2 tokens per round: a verbatim file rewrite decodes at 213 tok/s on
  the 27B (was 135) and 175 on FP4-Local (was 150), same tokens; prose is unchanged.
* **Prefix forking.** The server snapshots the state where the first turn after a system message begins, and
  a new conversation that starts with a system prompt another conversation has read gets that prefix copied
  into its slot (16-21 ms for 3,600 tokens, from a sibling on the GPU or a parked one) while the sibling keeps
  its own state: a fleet of agents with one system prompt reads it once. First turns of such siblings: 3-5.5 s ->
  0.15-0.2 s at 3,600 tokens of shared prompt.

### What the toks tokenizer changed (2026-10-19)

The server encodes prompts through [toks](https://github.com/actual-computer/toks), a C tokenizer library (AVX2
and NEON kernels, Apache-2.0) that returns exactly the ids Hugging Face `tokenizers` returns, loaded over a
`tokenizer.json` that bnk writes from the GGUF's own vocabulary (`serve/tokenizer_json.py`); without its wheel
the Python path runs as before, with the same ids (`tools/tok_check.py` checks both). The tokenizer itself is
18-33x faster on prose and code (a 256K-token prompt: 205 -> 10 ms), which takes 0.1-0.3 s off a cold 64K-256K
first turn and nothing off follow-up turns, whose encoding was already a few milliseconds. Each response's
`timings` now carries `tokenize_ms`. Details and the sweep on both models: `docs/TOKS.md`.

### What v0.4.0 changed: the dense Qwen3.8 models

A second engine (`src/dense/`, behind the `EngineApi` interface in `src/engine/engine_api.h`) runs the dense
`qwen35` models with every weight in VRAM; `bnk` picks the engine from the GGUF's architecture, so the
mixture-of-experts path is untouched (its greedy output is identical token for token, its speed within noise).
What the engine adds: tensor parallelism over the two cards (heads, DeltaNet groups and FFN rows split, partial
sums exchanged through pinned host memory since peer copies take 51 ms on this PCIe fabric), one CUDA graph
spanning both devices per window, a tensor-core flash-attention kernel for prompt chunks and a split-KV one for
decode windows (one pass over the cache per GQA group, 8-bit KV cells dequantized while staging), and the model's
own MTP block as the drafter. The DeltaNet output gate of this family is SiLU rather than sigmoid: the one place
the shared kernels gained a variant. Config `qwen27b` (`configs/Qwen3.8-27B-Q8_0.json`): 262K context, 8-bit KV,
two slots. Design, profiles and open items: `docs/DENSE.md`.

### What v0.3.1 changed: faster turns on both models

A performance sweep measured with the LocalMaxxing context sweep (`-c 16`), details in `docs/PERF-2026-10-16.md`:

| | FP4-Local v0.3.0 | v0.3.1 | IQ3_S v0.3.0 | v0.3.1 |
|---|---|---|---|---|
| 4K / 16K cold read | 1,186 / 1,935 tok/s | **1,337 / 2,067** | 1,948 / 2,430 | **1,990 / 2,465** |
| decode at 4K / 16K | 105 / 97 tok/s | **114 / 119** | 105-122 | **121 / 129** |
| follow-up turn wait at 4K / 16K | 1.2 / 1.2 s | **0.94 / 0.99 s** | 0.60 / 0.59 s | **0.56 / 0.58 s** |

* The n-gram PLE table is locked in RAM at load (`BNK_PLE_LOCK=0` only advises it). Its rows are read at hashed
  positions out of a 27-29 GB file mapping; a row whose page the cache had dropped was a disk read, which made a
  follow-up turn take 600 ms or 1,150 ms depending on what else had used the page cache. The service unit sets
  `LimitMEMLOCK=infinity`.
* The token embedding table lives in page-locked host memory (`BNK_EMBD_HOST=0` keeps it on the GPU) and a BF16
  output head becomes Q8_0 at load (`BNK_HEAD_Q8=0`): 1.2-1.9 GiB per card back to the expert cache. FP4-Local goes
  from 72% to 75 / 79% of its experts resident, so its reads copy less over PCIe and its decode runs fewer experts
  on the CPU.
* NVFP4 experts on the CPU use a multi-row AVX2 kernel, 1.45x ggml's per-token dot product, bit-identical to it.

Output on the IQ3_S release is unchanged token for token.

### What v0.3.0 changed: NVFP4 models on the V100s

NVIDIA's FP4 release of this model, [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
(ModelOpt NVFP4 experts, MXFP8 dense weights, quantization-aware distilled: GPQA 91.5 against 92.2 for BF16), runs
on the two V100s, cards whose silicon has no FP4 or FP8 arithmetic at all. A weight format is only bits: bnk decodes
NVFP4 (e2m1 codes with an e4m3 scale per 16 weights) to int8 or fp16 in registers inside its existing quantized
kernels, as it does for the IQ formats, and the products run on the V100's own fp16 tensor cores and `dp4a`. Every
weight is the checkpoint's, bit for bit: the routed experts (NVFP4, with their per-expert global scales), the
n-gram table (NVFP4, as a 16-wide block type), embeddings, routers and norms (BF16 / F32). The dense MXFP8
projections are stored as Q8_0, the one re-quantization. `docs/NVFP4.md` has the design and the measurements.

| Two V100s, 4K-token follow-up, drafting | FP4-Local (NVFP4) | IQ3_S release |
|---|---|---|
| Experts | 63.3 GiB, 74% resident per card, the rest on the CPU | 46.8 GiB, all resident |
| Decode, ranked expert cache | 93-99 tok/s | 105-120 tok/s |
| Drafts accepted | 41-44 of 49 | similar |

Converting it is two commands (`tools/convert_nvfp4.py`, which drives llama.cpp's converter and decodes the
checkpoint's compressed expert scales, and `tools/convert_ple_nvfp4.py` for the table; about 20 minutes, 70 GB of
RAM, 105 GB of disk for the two GGUFs) and the config `fp4-local`. Any NVFP4 GGUF of this architecture loads: the
format is in every path (decode, the tensor-core prompt path, the CPU experts, the VRAM cache).

### What the tensor-core prompt path changed (2026-10-15, two V100s)

The routed experts of a prompt chunk run on the tensor cores in one launch per layer (`src/kernels/moe_tc.cu`:
the quantized sub-blocks unpacked once per 64-token tile into fp16 fragments), instead of an fp16 copy of every
used expert through HBM and one small cuBLAS kernel per expert; long prompts read in 4,096-token chunks. The
LocalMaxxing context sweep (`/opt/benchmarks/bnk_bench.py`, batch 1, cold read then a ~512-token follow-up turn):

| depth | cold read, before -> after | follow-up turn wait, before -> after | decode |
|---|---|---|---|
| 4K | 1,283 -> **1,948 tok/s** | 729 -> **600 ms** | unchanged |
| 16K | 1,639 -> **2,430** | 743 -> **588** | unchanged |
| 32K | 1,861 -> **2,741** | 771 -> **603** | unchanged |
| 64K | 2,059 -> **2,874** | 838 -> **621** | unchanged |
| 128K | 1,955 -> **2,875** | 953 -> **752** | unchanged |
| 256K | 2,109 -> **2,653** | 1,160 -> **946** | unchanged |

Greedy output is unchanged token for token; `tests/test_moe_tc.cu` checks the kernel against a host reference
on every expert format. What was tried and not kept, and what is next: `docs/ROADMAP.md`, `docs/PLAN.md`.

### What v0.1.3 changed

A fix for agent workloads with more conversations than slots, where v0.1.2 could fail requests after a while with
`CUDA VMM create: out of memory`, then `not enough VRAM for a context of ...` or `prompt buffers: not enough VRAM`:

| | v0.1.2 | v0.1.3 |
|---|---|---|
| VRAM the budget believed free but the driver did not have (one agent session, ~60 parks/resumes) | ~1.5 GiB, growing | ~0.1 GiB, corrected at once |
| Requests failed for VRAM in that session | 3+ | 0 |

Batched layouts change all the time, so the engine replaces cached forward graphs often (it keeps 32). It credited
a destroyed graph's ~18 MiB back to the VRAM budget, but the driver keeps that memory for the next graph: each
replacement made the budget think it had 18 MiB more than it did, the expert cache grew into it, and the next
mapping failed. Destroyed graphs are no longer credited, and the budget now re-reads the driver's free memory
whenever it hands out VRAM: anything taken outside it (graphs, workspaces, staging) comes back out of the expert
cache, keeping the 0.75 GiB reserve. A failed mapping is unwound and retried once after reclaiming, and a VRAM
error now says why (`budget used / limit, driver free, reserve, cache gave`). The terminal's live line and the
dashboard's decode charts also show the **combined** speed of all requests decoding together (each request's own
rate counts the whole shared step, so with 3 at once each shows ~20 tok/s while the server delivers ~60).

### What v0.1.2 changed

| | v0.1.1 | v0.1.2 |
|---|---|---|
| An agent decoding while another agent's 42K-token first turn is read: its longest stall | 36.1 s | **6.7 s** |
| 3 conversations decoding together, drafting overhead per round | 7.99 ms | **6.04 ms** |
| ... their total decode speed (`bnk multi`, ~32-42K each, two runs each) | 80.8 / 85.7 tok/s | **86.9 / 90.9 tok/s** (+6-7%) |
| 3-agent server benchmark (3 turns of up to 1,500 tokens), overall | 34.2-35.6 tok/s | 33.1-35.5 tok/s (unchanged within noise) |

The first row is decoding between another request's prompt chunks; the reading request takes longer in exchange
(36 s of its own reading became 52 s of wall time: `BNK_READ_SHARE` sets the split). The next two rows are batched
drafting, which pays while several conversations decode together; in the closed-loop server benchmark the agents
are often at different stages, and the difference stays inside the run-to-run noise. v0.1.2 also fixes a crash
("expert staging too small") when a prompt was read while the expert cache had swaps in flight, which batched
decoding made likely.

### What v0.1.1 changed

| | Before | v0.1.1 |
|---|---|---|
| 3 agents at ~45-50K tokens each, follow-up turn read (`tools/multi_agent_bench.py`) | 41 s | **2.0 s** |
| ... the whole 3-agent run (4 turns each) | 721 s | **256 s** |
| A real coding agent's 3 workers: share of busy time spent reading prompts | 93% | **26%** |
| Decode, served (temperature 1.0, top-k 20), 3 agents at ~45-50K context | 54.0 tok/s | **61.9 tok/s** (+15%) |
| Decode at 78K context, drafting + commit overhead per round | 2.97 ms | **2.46 ms** (+1.3% decode) |
| Prompt attention kernel (8K rows at ~62-70K depth) | 88.8 ms | **50.1 ms** (1.77x) |
| Reading a 78K-token prompt | 63.3 s | **58.8 s** (+7%) |
| 3 conversations decoding at once, engine (`bnk multi`, ~32-42K each) | 65-68 tok/s total | **84 tok/s** (+24%) |
| 3 agents through the server, generation-heavy (3 turns of up to 1,500 tokens) | 31.2 tok/s overall | **35.7 tok/s** (+14%) |

The first two rows come from parking conversations in host RAM, the next three from moving sampling to the GPU
and overlapping the drafter with the commit, the attention rows from a rewritten prompt-attention kernel, and the
last two from batched decoding. How each was found and measured: `docs/ROADMAP.md`.

### Decoding

Measured before v0.1.1. Sampled decoding has since gained ~15% from sampling on the GPU (measured with three
agents at ~45-50K context, temperature 1.0); greedy decoding is unchanged by it.

| | IQ3_S (47 GB of experts) | Orca IQ4_XS (61 GB of experts) |
|---|---|---|
| Speculative, greedy, short context | **65 tok/s** | **71 tok/s** |
| Speculative, served (sampled), short context | ~58 tok/s | ~61 tok/s |
| Without speculation, greedy | 53 tok/s | 47 tok/s |
| Speculative at 120K tokens of context | 44 tok/s | 59 tok/s |
| Speculative at 228K tokens of context | 43 tok/s | 52 tok/s |

The long-context rows decode 300 tokens after a prompt of real source code (llama.cpp's), greedily, with the
models' speculation settings.

### Prompt processing

The same source-code prompts, read from scratch (measured before v0.1.1's attention kernel, which reads long
prompts ~7% faster):

| Prompt length | IQ3_S | Orca IQ4_XS |
|---|---|---|
| ~1.8K tokens | 338 tok/s | 290 tok/s |
| ~30K tokens | 685 tok/s | 628 tok/s |
| ~120K tokens | 714 tok/s (2.8 min) | 666 tok/s (3.0 min) |
| ~228K tokens | 922 tok/s (4.1 min) | 848 tok/s (4.5 min) |

Code routes to a wide spread of experts, so most of a chunk's experts stream over PCIe; text with narrower
routing processes faster (a synthetic 228K-token prompt ran at 1,094 tok/s).

### Long conversations and agents

The workload that matters most in practice. Logs of real use show agent-style sessions: a median context of
56-66K tokens (a quarter of requests above 100K), ~97% of each prompt shared with the previous turn, and far more
time spent generating than reading prompts once conversations are kept between turns.

One conversation (`tools/agent_bench.py`: the repository's own sources as an 80K-token context, then turns that
each add a tool result and a question; Orca, before v0.1.1):

| | Orca IQ4_XS at ~80K tokens of context |
|---|---|
| Decode, greedy | 65-74 tok/s |
| Decode, served (temperature 0.6) | 58 tok/s |
| Expert misses (served by the CPU) | 7-9% |
| Starting a follow-up turn | ~0.5 s for a short message, ~4 s for a 1K-token tool result |

Several agents at once (`tools/multi_agent_bench.py`: three agents with their own ~45-50K contexts, IQ3_S):

| | One at a time, no parking | Parking (v0.1.1) | Parking + batched decoding (v0.1.1) |
|---|---|---|---|
| Follow-up turn read | 41 s | 2.0 s | 1.9 s |
| 4 turns x 600 tokens each, wall time | 721 s | 256 s | |
| 3 turns x up to 1,500 tokens each, overall tok/s | | 31.2 | **35.7** |

Runs of these benchmarks vary by about ±5% between batches (the adaptive expert cache, GPU clocks, the VM's
host), so comparisons are made back to back and repeated.

For reference, the engine bnk replaces reached ~57 tok/s for decoding and 400-700 tok/s for prompt processing on
the same machine.

### Correctness

Logits match llama.cpp's reference implementation token by token: 44/44 argmax agreement and a mean KL divergence
of 0.003 on the validation prompt, comparable to the variation llama.cpp shows against itself across batch sizes.
Exactness checks that every change has to pass (static CPU expert tier, greedy):

* speculative decoding emits exactly the tokens of plain greedy decoding;
* a conversation parked in host RAM and brought back emits exactly the tokens it would have without the
  interruption (`tools/park_check.py`, also with several slots);
* each conversation decoded in a batch with others emits exactly the tokens it emits alone (`bnk multi`), and
  batched drafting proposes exactly the drafts each conversation's own drafter would;
* requests served concurrently, with prompts read between other requests' decoding rounds, emit exactly what
  they emit alone (`tools/serve_check.py`).

A needle hidden at 40% depth in a 228K-token prompt is retrieved verbatim.

## How it works

```
                 ┌────────────────────── V100 32 GB ───────────────────────┐
  tokens ──────► │ dense weights (attention, DeltaNet, hyper-connections)  │
                 │ VRAM expert cache: hottest experts, adaptive  ◄─┐ elastic budget
                 │ KV + indexer keys per conversation slot        ◄┤ (CUDA VMM)
                 │ prompt buffers: borrowed for long prompts      ◄┘
                 └──────┬───────────────────────────────▲──────────────────┘
               misses   │ mapped-memory mailbox          │ PCIe streaming (prompts)
                        ▼                                │
                 ┌─────────── host: 128 GB RAM, 24 threads ───────────────┐
                 │ every expert, page-locked · AVX2 expert kernels         │
                 │ parked conversations (pinned)                           │
                 └─────────────────────────────────────────────────────────┘
```

**Decoding.** Each step runs the whole layer stack as one captured CUDA graph. The router's choices are
deduplicated on the GPU; experts resident in VRAM run there, and the misses for the layer are posted to a
host-mapped mailbox that a pool of 24 CPU threads answers while the GPU carries on. The CPU kernels are bnk's own
AVX2 code (bit-identical to ggml's dot products) that decode each weight row once for all the tokens of a
speculative window. The expert cache starts from the routing counts learned in earlier sessions and keeps
adapting: it swaps the hottest non-resident experts in, asynchronously, on a copy stream. Sampling (temperature,
top-k, top-p, min-p) picks each row's top-k candidates on the GPU, so only k values per row cross PCIe.

**Weights on the GPU.** Quantized matrices (Q4_K-Q8_0, IQ2_XXS-IQ4_XS, Q2_0, BF16) are repacked at load into an
"R layout": a lossless struct-of-arrays rearrangement of each row's blocks so every warp issues coalesced 16-byte
loads. GEMVs use dp4a, with activation sums that fold in the zero offsets; small batches use split-K.

**Speculative decoding.** The model's multi-token-prediction layer drafts up to four tokens per step, stopping
early when its confidence drops below a cutoff. The main model then verifies the whole window in one pass, and
the DeltaNet recurrent state is replayed for the accepted prefix, on a second stream while the drafter prepares
the next window. With several conversations decoding together, every conversation's drafter pass runs in one
forward (each on its own drafter KV, the projections, MoE and head shared). Sampled requests use exact rejection sampling, so the output distribution is the model's own.
The drafter scores only a ~145K-token subset of the vocabulary (Latin scripts, code, symbols, emoji) and, like the
main model, attends sparsely through its own indexer.

**Several conversations at once.** The engine holds `--slots` conversations on the GPU (3 in the server by
default), each with its own KV cache, indexer keys, DeltaNet state, snapshots and drafter KV. One forward runs
every active conversation's verify window together: the row-wise work (norms, projections, the MoE, the head) once
over all rows, attention, DeltaNet and PLE per conversation on its own state. The rows of a step (8 at most) go
first to each conversation's pending token, then to drafts in turn. Mixed conversations share one expert cache and
miss more, so with several slots the cache adapts every step. Each batch layout is its own CUDA graph (~18 MiB),
kept in a small LRU charged to the VRAM budget.

**Long context.** The full-attention layers use the model's QSA sparse attention: a learned indexer scores
compressed 4-token blocks and each query attends to its top 2,048 tokens. bnk computes the block scores as a tiled
fp32 product over 64 queries at a time; the attention then runs on tensor cores in one launch over the whole
chunk, one block per (row, KV head) serving the twelve query heads that share it, with the next tile's gathered
keys and values loaded while the current one computes. Every query sees a fixed number of keys, so prompt
processing does not slow down as the context fills; streaming experts over PCIe dominates it, with attention
about 10% of the GPU time of an 80K-token read (`tests/bench_qsa_attn.cu` times the kernels on real selections).

**Elastic VRAM.** The KV caches, the indexer keys, the drafter's KV and the prompt-processing buffers each reserve
address space for their maximum and map physical memory in 2 MiB pages only as needed. All of them draw on one
budget whose remainder is the expert cache. When a conversation grows or a long prompt arrives, the cache gives up
its coldest slots (moving hot experts out of the freed pages first); afterwards it grows back. A 256K context
therefore costs nothing until it is used. `BNK_KV8=1` stores the K/V cells in 8 bits (a fp16 scale per 32 values,
as q8_0) instead of fp16: 9 instead of 14 KiB per token per GPU, so 256K tokens take 2.2 rather than 3.6 GiB of
each card's expert cache, and a parked conversation 25-40% less RAM. Its effect on the output is within what
reordering the kernels' sums does (`bnk nll` with `BNK_NLL_REF`: KL 0.015 from fp16 at 64K tokens, against
0.011-0.012 for the older attention and expert kernels).

**Prompt processing.** Prompts are processed in chunks (2K tokens, or 8K for long prompts) with tensor-core
GEMMs: resident experts run as grouped GEMMs, non-resident ones stream over PCIe one layer ahead into
double-buffered staging, and the CPU takes the experts that are cheaper to compute there.

**Conversations.** Chat clients re-send the whole conversation each turn and usually re-render earlier turns
(Qwen's template drops a previous answer's reasoning), so a new prompt shares only part of what the engine has
already processed. The KV cache is stored per position and stays valid, but the DeltaNet layers carry running
state, so bnk snapshots that state where each prompt's last turn begins; the next request rewinds to the latest
snapshot inside the shared prefix and processes only what is new. When more conversations take turns than there
are slots, the engine **parks** a conversation's whole state (history, KV, indexer keys, recurrent state,
snapshots, the drafter's KV: 28 KiB per token) in pinned host RAM and brings it back in ~0.15 s for its next turn,
instead of reading its prompt again (`--park-gib`, 24 GiB by default, least recently used dropped first). A
request that has generated for `--slice` seconds (10) while others wait for a slot gives its turn up and continues
where it was, so one long answer does not hold the other agents up.

**Several GPUs.** `--gpus` (the server's default: every GPU) splits the layers into contiguous ranges, one per GPU,
balanced by the memory each needs and has. Each GPU holds its layers' dense weights, experts, KV and recurrent
state; the first one embeds the tokens, the last one holds the head and the drafter; only the residual rows
(40 KiB per token) cross between them, through pinned host memory. While one GPU reads a prompt chunk, the next
GPU reads the previous chunk. One copy of the experts in host RAM and one CPU pool serve both. Details,
projections and how it was checked on one card: `docs/DUAL-GPU.md`.

**Thinking-loop guard.** Reasoning models occasionally loop. The server watches the thinking tokens and, when
most of a recent 4K-token window repeats earlier 12-token stretches, lets the model finish its sentence, closes
the reasoning with a short note, and lets the model answer within the remaining token budget. The hand-over
reuses the engine's state, so it costs ~0.3 s regardless of the conversation's length. Real reasoning traces stay
under 5% repetition, well below the 55% trigger.

## Models

bnk runs any GGUF of the `qwen4exp` architecture (Qwen3.8-Flash-Next and fine-tunes): 48 layers (36 gated DeltaNet
+ 12 QSA full-attention), 512 routed experts (top 10) plus a shared expert, hyper-connections (4 streams), the
per-layer n-gram embedding. Expert formats: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q2_0, Q4_K,
Q5_K, Q6_K, Q8_0, NVFP4 (with llama.cpp's per-expert `.scale` tensors; `docs/NVFP4.md`). Files that also carry an
MTP block (`nextn_predict_layers`) load their trunk.

It also runs GGUFs of the `qwen35` architecture (the dense Qwen3.8 models: 27B) through the dense engine of
`src/dense/` (`docs/DENSE.md`): every weight in VRAM, any quant the R layouts support (Q8_0 is the one measured),
the model's own MTP block as the drafter.

Each model is one file in `configs/`: its settings and the Hugging Face files it comes from. `./run.sh` lists
exactly these models, and `./models.sh` manages them:

```bash
./models.sh list                               # the configured models, and which are downloaded
./models.sh download iq3_s                     # download one (resumes if interrupted, checks every SHA-256)
./models.sh download iq3_s --dir /data/models  # ... into a folder of your choice (remembered in bnk.env)
./models.sh add SomeOrg/Some-Model-GGUF --include 'IQ4_XS/*' --alias some
                                               # add a model: writes configs/<name>.json, then download it
```

| Config | Model | Size |
|---|---|---|
| `iq3_s` (default) | [ISTA-DASLab Qwen3.8-Flash-Next-GSQ-RCO IQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 84 GB, 2 shards |
| `cyber-frost` | [peasantsmith CYBER-FROST-3.8-PS-GUFF Q5_K_M](https://huggingface.co/peasantsmith/CYBER-FROST-3.8-PS-GUFF) (Blackfrost-AI's cybersecurity fine-tune) | 112.5 GB, 1 file; needs ~110 GB of RAM |
| `fp4-local` | [nvidia Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) (NVFP4, quantization-aware distilled), converted locally with `tools/convert_nvfp4.py` and `tools/convert_ple_nvfp4.py` (`docs/NVFP4.md`) | 75 GB + 29 GB table, two local GGUFs; the config's paths point at them |
| `qwen27b` | [unsloth Qwen3.8-27B-GGUF Q8_0](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (the dense 27B; `docs/DENSE.md`) | 29 GB, 1 file; needs two V100s |

**Where models go.** By default into the Hugging Face cache (`~/.cache/huggingface/hub`, or `$HF_HOME` /
`$HF_HUB_CACHE`), the same place `hf download` uses, so a model already downloaded with `hf` is found as it is. With
`--dir DIR` (or `./setup.sh --models DIR`) models go to `DIR/<config name>/` instead; the folder is saved to
`bnk.env` as `BNK_MODELS`, and later downloads and `run.sh` use it. A model is looked for in `BNK_MODELS` first, then
in the cache.

**Adding a model.** `./models.sh add REPO` reads the repo's file list; if it holds several models (quantization
levels), it lists them with the `--include` pattern for each. The new config pins the repo's current commit and
starts from bnk's default sampling and speculation (see [Per-model configs](#using-it)); tune them there. A model
downloaded some other way can be configured by hand with `"source": {"path": "/path/to/model-00001-of-0000N.gguf"}`.
`./run.sh /path/to/model-00001-of-0000N.gguf` also runs any GGUF without a config.

**Gated models** need a Hugging Face account that accepted the model's terms on its page, and a read token
(https://huggingface.co/settings/tokens) on this machine: `.venv/bin/hf auth login`.

## Setup in detail

### What you need

| | |
|---|---|
| OS | Linux, x86_64. `setup.sh` installs the driver and CUDA itself on **Ubuntu 22.04 or 24.04**; on other distributions install them by hand first. |
| GPU | an sm_70 GPU (V100). Others are untested and would need their own tuning. |
| Driver | NVIDIA 570 or newer (CUDA 12.8 needs it). |
| CUDA | **12.x** (12.8 tested). CUDA 13 cannot build for the V100. |
| RAM | enough to page-lock every expert, plus ~10 GB: 64 GB for IQ3_S (47 GB of experts). |
| CPU | AVX2 (any recent x86_64). |
| Disk | ~90 GB per model, ~6 GB for the MTP layer and its source tensors. |
| Network | for the driver, the packages, the MTP layer and the models. |

### Step by step

1. **Get the code.**
   ```bash
   git clone https://github.com/christ-pher/bnk.git
   cd bnk
   ```
2. **Run the setup.**
   ```bash
   ./setup.sh --mtp
   ```
   It works through six steps and asks (`[Y/n]`, Enter means yes) before anything that changes the system:
   1. **The machine:** GPU, driver, RAM, AVX2. Without a working NVIDIA driver (and on Ubuntu) it offers to add
      NVIDIA's repository and install the driver with CUDA 12.8 (`cuda-drivers cuda-toolkit-12-8`). It then
      stops and asks for a **reboot**. If Secure Boot is on, it warns you first: see
      [Troubleshooting](#troubleshooting).
   2. **System packages:** compiler, git, curl, Python venv, whiptail (with `apt`, using `sudo`), and the CUDA
      12.8 toolkit if the driver is there but the toolkit is not.
   3. **The Python environment** (`.venv`): the server's dependencies, CMake and Ninja if the system's are too old.
   4. **The engine** (`build/bnk`), compiled for the V100. The log is `build-cmake.log`.
   5. **The dashboard** (`serve/web/dist`), with a bundled Node.js if there is none.
   6. **Models and the MTP layer:** if the default model is not downloaded yet, it offers to download it (84 GB)
      into the Hugging Face cache, or into `--models DIR` (saved to `bnk.env`, which `run.sh` reads). With `--mtp`
      it builds the speculative-decoding layer from the official Qwen checkpoint (~5 GB downloaded, ~25% faster
      decoding) into `~/.cache/bnk/mtp/`.
3. **If it asked you to reboot:** `sudo reboot`, log back in, `cd bnk` and run `./setup.sh --mtp` again. It picks
   up where it stopped.
4. **Start the server.**
   ```bash
   ./run.sh
   ```
   and pick the model from the list (or name it: `./run.sh iq3_s`).
   The first start takes about a minute (the experts are copied into page-locked RAM and the VRAM cache filled).
   It is ready when the terminal shows `listening on http://0.0.0.0:8080`.
5. **Open the dashboard** at `http://<the machine's IP>:8080` (`hostname -I` shows the IP), or send a request:
   ```bash
   curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"Hello!"}]}'
   ```

Everything at once, without questions (the driver install still needs the reboot and a second run):

```bash
./setup.sh --yes --mtp --download iq3_s
```

| `setup.sh` option | |
|---|---|
| `--yes` | do not ask; install what is missing |
| `--mtp` | build the MTP draft layer (recommended) |
| `--models DIR` | a folder for models (default: the Hugging Face cache; saved to `bnk.env`) |
| `--download iq3_s` | download these models (config names or aliases, comma-separated) |
| `--no-download` | never offer a download |
| `--no-system` | never touch system packages, only report |
| `--cuda DIR` | build with this CUDA toolkit |

**More models later:** `./models.sh list`, `./models.sh add`, `./models.sh download` (see [Models](#models)).

**By hand** (what `setup.sh` runs, for other distributions):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 && cmake --build build -j
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
(cd serve/web && npm ci && npm run build)
.venv/bin/python tools/build_mtp.py --out ~/.cache/bnk/mtp/mtp-q2_0.gguf
./models.sh download iq3_s
```

The cache ranking learned while serving is saved to `~/.cache/bnk/counts-<model>.bnkc` and used next time, so the
expert cache starts warm after the first session.

## Using it

**run.sh**

| | |
|---|---|
| `./run.sh` | on a terminal: pick a model from the list of `configs/` (a model not downloaded yet is offered for download) |
| `./run.sh iq3_s` | a model by its config name or alias, or a path to a model's first `.gguf` shard |
| `--port 8080` | HTTP port |
| `--ctx 262144` | maximum context (256K is the model's native length; memory is only used as needed) |
| `--slots 3` | conversations decoded at once (batched); `1` serves one at a time |
| `--no-mtp` | no speculative decoding |
| anything else | passed to the server (`.venv/bin/python -m serve.server --help`); to keep an option, put it in the model's `server_args` |

| Environment | |
|---|---|
| `BNK_SLOTS=3` | conversations decoded at once |
| `BNK_KV8=1` | 8-bit K/V cache instead of fp16 (40% less VRAM and parking RAM per token; output within kernel-reordering noise) |
| `BNK_PARK_GIB=24`, `BNK_SLICE_S=10` | host RAM for parked conversations, and the time slice when requests wait for a slot; `0` turns either off |
| `BNK_READ_SHARE=0.5` | while a prompt is read, the other requests decode for this share of each chunk's time (`0`: reads block them) |
| `BNK_DRAFT_BATCH=0` | draft for each conversation on its own instead of in one batched pass |
| `BNK_LOG_LEVEL=quiet\|info\|debug` | terminal output: `info` prints a line per request and a live status line |
| `BNK_VRAM_LOG=1`, `BNK_VMEM_LOG=1` | log to the engine log when the VRAM budget corrects itself against the driver / when the expert cache shrinks or grows |
| `BNK_THINK_GUARD=0` | turn the thinking-loop guard off |
| `BNK_DRAFT_VOCAB=` | draft over the whole vocabulary (e.g. for chats in non-Latin scripts) |
| `BNK_MODELS`, `BNK_MTP` | a models folder (instead of the Hugging Face cache) and the MTP layer (`setup.sh` and `models.sh` save them to `bnk.env`) |
| `BNK_API_KEY` | require this key from API clients |
| `BNK_CPU_PIN=1`, `BNK_CPU_CHUNKS=3` | pin the CPU expert workers to cores / split their work dynamically (no measurable gain on this machine, so off by default) |

**Per-model configs** - `configs/<name>.json`:

```json
{
  "notes": "free text",
  "aliases": ["iq3_s"],
  "source": {
    "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
    "revision": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
    "files": ["IQ3_S/...-00001-of-00002.gguf", "IQ3_S/...-00002-of-00002.gguf"]
  },
  "sampling": { "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0 },
  "speculation": { "draft": 4, "draft_min_p": 0.8 },
  "thinking_loop_guard": true,
  "server_args": ["--model-id", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S", "--max-tokens", "32768", "--slots", "3"],
  "engine_args": ["--threads", "24"]
}
```

This is where a model's persistent settings go; the file's name is the model's name, and `run.sh` takes it or
one of its `aliases`. `source` says what `./models.sh download` fetches (every shard, the first one first, at the
pinned `revision`; or `"path"` for a model already on disk). For a `.gguf` path, `run.sh` loads the config named
after the file if there is one. `server_args` are any server options (port, context, slots, model id,
completion budget, API key ...) and are read as if typed before the command line, so options given to `run.sh`
still win. `engine_args` go to the engine (`--threads`, `--cache-gib`, ...; see `tools/bnk_main.cpp`). Sampling
values are defaults; a request's own fields take precedence. The shipped speculation settings come from sweeps on
this machine: drafting up to 4 tokens with a 0.8 confidence cutoff was 13-18% faster than the engine's defaults
(3 tokens, 0.5 cutoff). Sampling with repetition or presence penalties, or with `top_k` 0, runs on the CPU (slower,
especially `top_k` 0); everything else picks its candidates on the GPU.

**APIs** - OpenAI `POST /v1/chat/completions` and `/v1/completions` (streaming, tools, `reasoning_content`),
Anthropic `POST /v1/messages`, `GET /v1/models`. Thinking is on by default; turn it off per request with
`"chat_template_kwargs": {"enable_thinking": false}`. Requests may also set `draft` and `draft_min_p`.

```bash
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

**Dashboard** - live analytics built with [shadcn/ui](https://ui.shadcn.com/): decode and prompt-processing
speeds, speculation, expert-cache hit rates per layer, GPU / PCIe / CPU / memory, **conversations** (who is
generating, who waits, which are on the GPU or parked in RAM, each one's turns), requests, the engine log, and a
chat panel for testing. The sidebar lists the OpenAI and Claude API base URLs and the served model id. Data arrives
over server-sent events (`GET /api/stream`); `GET /api/stats` returns a snapshot.

**Several agents.** Up to `--slots` requests (3) run at once, decoded together; more wait in line and take turns
by time slice. Every conversation keeps its state between turns, on the GPU while it fits a slot and parked in
host RAM otherwise, so a follow-up turn reads only its new tokens. While one request's prompt is read, the others
keep decoding between its chunks (half of each chunk's time by default, `BNK_READ_SHARE`): a long first turn no
longer freezes every other agent.

**Engine CLI** - `build/bnk` also runs on its own:

```bash
build/bnk run   --model M --tokens-file ids.csv --max-new 200 --mtp mtp.gguf   # generate, with timings
build/bnk multi --model M --slots 3 --tokens-file a.csv,b.csv,c.csv --max-new 200 [--solo]
                                                                               # several conversations batched
build/bnk check --model M --tokens-file ids.csv --ref ref/prefix              # vs llama.cpp, layer by layer
build/bnk pdump --model M --tokens-file ids.csv --ref out.bin                 # process a prompt, write its logits
```

`BNK_SPLIT=N` makes `pdump` time only the tokens after the first N (a follow-up turn at depth),
`BNK_NLL_SKIP=N` makes `nll` read the first N tokens as a prompt and score the rest, `BNK_NLL_DUMP=file` /
`BNK_NLL_REF=file` save a run's top-64 log-probabilities / report another run's KL divergence from them,
`BNK_PROFILE_DECODE=1` limits an `nsys --capture-range=cudaProfilerApi` profile of `bnk run` to decoding, and
`BNK_ROUTE_LOG=file` records every decode step's routing and timing for `tools/batch_sim.py`.

**Tools and tests:** `tools/multi_agent_bench.py` and `tools/agent_bench.py` (agent workloads against a running
server), `tools/park_check.py` (parking exactness), `tools/serve_check.py` (concurrent serving exactness, reads
interleaved with decoding), `tools/llama_ref.cpp` (llama.cpp reference dumps for `check`),
`tests/` (kernel tests: `test_dequant`, `test_cpu_kernels`, `test_topk`; benchmarks: `bench_qsa_attn`,
`bench_sampler`, `bench_gemv`, `bench_cpu_experts`).

## Troubleshooting

| Problem | What to do |
|---|---|
| `setup.sh`: "no working NVIDIA driver" on a distribution other than Ubuntu 22.04/24.04 | install the driver (570+) and the CUDA 12.8 toolkit from NVIDIA's site, reboot, run `./setup.sh` again |
| After installing the driver, `nvidia-smi` fails | reboot. With **Secure Boot** on, the first boot after the install shows a blue "Perform MOK management" screen: choose *Enroll MOK*, *Continue*, type the password the install asked for, reboot. If you missed it: `sudo mokutil --import /var/lib/shim-signed/mok/MOK.der`, reboot, enroll. Or turn Secure Boot off in the firmware. |
| "driver ... is older than CUDA 12.8 needs" | `sudo apt install cuda-drivers`, reboot |
| "CUDA 13 cannot target the V100" | install `cuda-toolkit-12-8` next to it; `setup.sh` finds `/usr/local/cuda-12.8` (or pass `--cuda /usr/local/cuda-12.8`) |
| The build fails | the end of `build-cmake.log` names the error; a compiler older than g++ 12 is the usual cause |
| "this model is gated" | accept the terms on its Hugging Face page, `.venv/bin/hf auth login`, run the download again |
| "not enough disk space" | free some space, or put the models on a bigger disk: `./models.sh download NAME --dir DIR` |
| A download stopped | run the same command again: finished files are kept, checked files are not re-hashed |
| The server exits at start with an out-of-memory error | not enough RAM for the model's experts (see What you need); close other programs or use a smaller quantization |
| `no model '...' in configs/` from `run.sh` | `./models.sh list` shows the names and aliases; `./models.sh add REPO` adds a model |
| `... is not downloaded yet` | `./models.sh download NAME`; if it is in a folder of your own, set `BNK_MODELS` in `bnk.env` (files at `DIR/<name>/...`) |
| Port 8080 in use | `./run.sh iq3_s --port 8081` |
| The dashboard is missing (the APIs still work) | run `./setup.sh` again; it rebuilds `serve/web/dist` |

## Repository

```
src/core/      GGUF loading, model config, expert cache, elastic VRAM (vmem)
src/kernels/   CUDA: R-layout GEMV, MoE, QSA sparse attention, DeltaNet, sampling (top-k), GEMM helpers
src/cpu/       expert store, CPU expert pool, AVX2 multi-token kernels
src/engine/    engine (decode, batched decode, prefill, parking), MTP drafter, generator (speculation, sampling);
               engine_api.h: the interface both engines implement
src/dense/     the dense engine (qwen35: Qwen3.8-27B): weights in VRAM, pipeline or tensor parallel, its MTP block
src/server/    the engine's JSON-lines protocol and request scheduler (bnk serve)
serve/         HTTP server: APIs, chat templates, tokenizer, conversations, telemetry, loop guard, console
serve/web/     dashboard (React + Tailwind + shadcn/ui)
configs/       the models: settings and download source, one file each (models.sh manages them)
docs/          ROADMAP.md: what was measured, what changed, what is next; DUAL-GPU.md: several GPUs; DENSE.md: the dense engine
tools/         CLI, models.py (behind models.sh), MTP builder, draft-vocabulary builder, agent benchmarks, batching simulator, checks
tests/         kernel tests and benchmarks
third_party/   ggml (CPU backend: quantization formats, GGUF), MIT
```

## Limitations

* Batched decoding shares at most 8 rows per step between the conversations, so each one verifies fewer drafts
  than alone (3 conversations: ~2.7 rows each).
* A prompt being read shares the GPU with the others' decoding, so it takes longer while others are active.
* Decoding slows as the context fills: the KV cache takes VRAM from the expert cache (up to 7 GiB per conversation
  at 256K, 4.3 GiB with `BNK_KV8=1`).
* The thinking-loop guard catches exact repetition, not paraphrased loops.
* Only sm_70 is built and tested; other GPUs would need their own tuning (and CUDA 13 no longer targets V100).
* With several GPUs the layers run on one GPU after the other: one conversation's decode does not get faster than
  one GPU holding everything would be; the gain is that everything fits. Two GPUs were tested as two stages on one
  card (exact against one GPU), not yet on two cards.

What is planned next, with measurements behind each item: `docs/ROADMAP.md`.

## Credits

The model architecture and weights are Qwen's; the IQ3_S quantization is ISTA DASLab's, and the Uncensored IQ4_XS
measured above orcarouter's. bnk vendors
[ggml](https://github.com/ggml-org/ggml) (MIT) for its CPU quantization formats and GGUF parser, and uses
[llama.cpp](https://github.com/ggml-org/llama.cpp) as its numerical reference.
