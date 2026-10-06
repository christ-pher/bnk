# bnk

A from-scratch inference engine for **Qwen3.8-Flash-Next-class mixture-of-experts models** (`qwen4exp`) on a
single **NVIDIA V100** with lots of host RAM. The model does not fit in VRAM, so bnk splits it across tiers: the
dense weights and the hottest experts live on the GPU, every expert lives in pinned system RAM, misses are computed
by the CPU during decoding, and experts stream over PCIe while a prompt is read.

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
[Supported models](#supported-models) · [Setup in detail](#setup-in-detail) · [Using it](#using-it) ·
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
./run.sh iq3_s
```

and open `http://<this machine's IP>:8080` for the dashboard. The OpenAI-compatible API is at `/v1`, the
Anthropic one at the root. The first start takes about a minute. Details, options and problems:
[Setup in detail](#setup-in-detail) and [Troubleshooting](#troubleshooting).

## Performance

All numbers were measured on the machine above. "Greedy" runs decode deterministically with a warm expert cache;
"served" runs go through the HTTP server with each model's default sampling (`configs/`), which is what a chat
client sees.

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

### CYBER-FROST-3.8 (Q5_K_M)

Measured back to back with Orca on the same prompts (`bnk run`, greedy, warm routing counts, each model's
speculation settings; the short-context rows average three chat prompts, the others decode 300 tokens after
llama.cpp source code; before v0.1.1):

| | CYBER-FROST Q5_K_M (78 GB of experts) | Orca IQ4_XS, same session |
|---|---|---|
| Speculative, greedy, short context | 52 tok/s | 58 tok/s |
| Without speculation, greedy | 40 tok/s | 47 tok/s |
| Speculative after ~1.7K / 30K tokens | 48 / 52 tok/s | 63 / 64 tok/s |
| Speculative at 120K / 228K tokens | 46 / 42 tok/s | 57 / 54 tok/s |
| Prompt processing, ~1.7K / 30K tokens | 212 / 627 tok/s | 248 / 697 tok/s |
| Prompt processing, ~120K tokens | 674 tok/s (3.0 min) | 740 tok/s (2.7 min) |
| Prompt processing, ~228K tokens | 742 tok/s (5.1 min) | 844 tok/s (4.5 min) |
| 80K agent conversation, greedy (`tools/agent_bench.py`) | 53 tok/s, 4.7 s per turn read | |
| 80K agent conversation, served (temperature 1.0) | 43 tok/s, 4.7-5.6 s per turn read | |

Its experts are 28% larger than Orca's, so fewer fit in VRAM (27% of them vs 34%): more of each token's experts
run on the CPU, which is bound by RAM bandwidth, and decoding is 10-22% slower (more so at long context).
Speculation at temperature 1.0 gains most from a high draft cutoff (draft 5 / cutoff 0.8: ~47.5 tok/s served at
short context, against ~41 with the engine defaults 3 / 0.5).

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
therefore costs nothing until it is used.

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

**Thinking-loop guard.** Reasoning models occasionally loop. The server watches the thinking tokens and, when
most of a recent 4K-token window repeats earlier 12-token stretches, lets the model finish its sentence, closes
the reasoning with a short note, and lets the model answer within the remaining token budget. The hand-over
reuses the engine's state, so it costs ~0.3 s regardless of the conversation's length. Real reasoning traces stay
under 5% repetition, well below the 55% trigger.

## Supported models

Any GGUF of the `qwen4exp` architecture (Qwen3.8-Flash-Next and fine-tunes): 48 layers (36 gated DeltaNet + 12
QSA full-attention), 512 routed experts (top 10) plus a shared expert, hyper-connections (4 streams), the
per-layer n-gram embedding. Expert formats: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q2_0, Q4_K,
Q5_K, Q6_K, Q8_0. Files that also carry an MTP block (`nextn_predict_layers`) load their trunk.

| Preset (`./run.sh <preset>`) | Model | Size | Download |
|---|---|---|---|
| `iq3_s` (default) | [ISTA-DASLab Qwen3.8-Flash-Next-GSQ-RCO IQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 84 GB, 2 shards | `tools/fetch_models.py iq3_s` |
| `orca` | [orcarouter Qwen3.8-Flash-Next-Uncensored IQ4_XS](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) (gated) | 98 GB, 3 shards | `tools/fetch_models.py orca` |
| `abliterated` | [SC117 GSQ-RCO-abliterated IQ3_S](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) | 84 GB, 2 shards | `tools/fetch_models.py abliterated` |
| `swift`, `swift-abliterated` | ukisai's Swift 1.5 GSQ-RCO IQ3_XXS, and [SC117's abliterated transplant](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) | | by hand, below |
| `cyber-frost` | Blackfrost-AI's CYBER-FROST-3.8 as `peasantsmith/CYBER-FROST-3.8-PS-GUFF` (Q5_K_M) | 84 GB without its PLE table | by hand, below |

`tools/fetch_models.py` downloads a preset into the folder `run.sh` expects, pinned to the revision bnk was tested
with, and checks every file's SHA-256 (`--list` shows the presets; run again to resume an interrupted download).
Any other `qwen4exp` GGUF runs with `./run.sh /path/to/model-00001-of-0000N.gguf`; add a `configs/<name>.json` to
set its sampling defaults.

**GSQ-RCO-abliterated IQ3_S.** The IQ3_S above with 144 residual-writing tensors swapped for Orca's abliterated
ones (mostly Q8_0 now); shard 2 and the MTP head are the stock ones. It starts from Orca's settings (temperature
0.6, guard on).

**Swift 1.5 GSQ-RCO-abliterated IQ3_XXS.** The same transplant applied to ukisai's Swift 1.5 IQ3_XXS: 144
residual-writing tensors replaced (IQ4_XS / IQ4_NL / Q2_0), the other 1080 byte-identical. Swift's shard 2 holds
layers 13-47, so both shards are needed. Card sampling (temperature 1.0).

```bash
.venv/bin/hf download SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF --include "IQ3_XXS/*" \
    --local-dir /opt/models/swift-gsq-rco-abliterated
./run.sh swift-abliterated   # BNK_SWIFT_ABLITERATED overrides the path; BNK_SWIFT for ./run.sh swift
```

**CYBER-FROST-3.8.** Of the published quantizations of this fine-tune, PS-GUFF has the most precise experts that
fit 128 GB of RAM and needs no new kernels (the other GGUFs use Q5_0/Q5_1/Q3_K/MXFP4 experts, or need more RAM).
Three things about the file:

* Its 28.8 GB PLE table is byte-identical to Orca's (checked with a SHA-256 of the whole tensor), so it is
  downloaded without it (`tools/fetch_gguf.py`, 83.7 GB instead of 112.5 GB) and the engine reads the table from the
  Orca file (`--ple-gguf`; `run.sh cyber-frost` passes it). Orca has to be downloaded too.
* It declares its MTP block (`block_count` 49, `nextn_predict_layers` 1); the engine runs the 48-layer trunk and
  drafts with `--mtp` as usual. That MTP head is byte-identical to the official Qwen one, so the stock draft layer
  from `tools/build_mtp.py` is the model's own.
* Its `compress_ratios` are all 0, which would mean dense attention, although the attention layers carry their
  QSA indexers and the checkpoint's config says `indexer_compress_ratio` 4; the engine restores the ratio
  (`BNK_NO_QSA=1` runs dense attention).

Its chat template adds Blackfrost's own system prompt and a reasoning-effort line (`xhigh` by default; pass
`chat_template_kwargs: {"reasoning_effort": "medium"|"low"}`), and always thinks.

```bash
.venv/bin/python tools/fetch_gguf.py --repo peasantsmith/CYBER-FROST-3.8-PS-GUFF \
    --file CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf --drop per_layer_token_embd.weight \
    --out /opt/models/cyber-frost/CYBER-FROST-3.8-PS-Q5_K_M-noPLE.gguf
./run.sh cyber-frost      # BNK_CYBER_FROST / BNK_PLE_GGUF override the two paths
```

## Setup in detail

### What you need

| | |
|---|---|
| OS | Linux, x86_64. `setup.sh` installs the driver and CUDA itself on **Ubuntu 22.04 or 24.04**; on other distributions install them by hand first. |
| GPU | an sm_70 GPU (V100). Others are untested and would need their own tuning. |
| Driver | NVIDIA 570 or newer (CUDA 12.8 needs it). |
| CUDA | **12.x** (12.8 tested). CUDA 13 cannot build for the V100. |
| RAM | enough to page-lock every expert, plus ~10 GB: 64 GB for IQ3_S (47 GB of experts), 80 GB for Orca (61 GB), 96 GB for CYBER-FROST (78 GB). |
| CPU | AVX2 (any recent x86_64). |
| Disk | ~90 GB per model, ~5 GB for the MTP layer. |
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
   6. **The models folder and the MTP layer:** the folder defaults to `~/models` (`--models DIR` to change it) and
      is saved to `bnk.env`, which `run.sh` reads. With `--mtp` it builds the speculative-decoding layer from the
      official Qwen checkpoint (~5 GB downloaded, ~25% faster decoding). If no model is there yet, it offers to
      download the default one (84 GB).
3. **If it asked you to reboot:** `sudo reboot`, log back in, `cd bnk` and run `./setup.sh --mtp` again. It picks
   up where it stopped.
4. **Start the server.**
   ```bash
   ./run.sh iq3_s
   ```
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
| `--models DIR` | where models go (default `~/models`; saved to `bnk.env`) |
| `--download iq3_s,orca,abliterated` | download these models (84 / 98 / 84 GB) |
| `--no-download` | never offer a download |
| `--no-system` | never touch system packages, only report |
| `--cuda DIR` | build with this CUDA toolkit |

**More models later:** `.venv/bin/python tools/fetch_models.py --list`, then for example
`.venv/bin/python tools/fetch_models.py orca --models ~/models`. Orca is gated: open its page on Hugging Face,
accept its terms while logged in, create a read token at https://huggingface.co/settings/tokens, and run
`.venv/bin/hf auth login` once.

**By hand** (what `setup.sh` runs, for other distributions):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 && cmake --build build -j
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
(cd serve/web && npm ci && npm run build)
.venv/bin/python tools/build_mtp.py --out ~/models/mtp/mtp-q2_0.gguf
echo 'BNK_MODELS="'$HOME'/models"' >> bnk.env; echo 'BNK_MTP="'$HOME'/models/mtp/mtp-q2_0.gguf"' >> bnk.env
```

The cache ranking learned while serving is saved to `~/.cache/bnk/counts-<model>.bnkc` and used next time, so the
expert cache starts warm after the first session.

## Using it

**run.sh**

| | |
|---|---|
| `./run.sh` | on a terminal: pick a model from a list |
| `./run.sh iq3_s` | a preset (`iq3_s`, `orca`, `abliterated`, `swift`, `swift-abliterated`, `cyber-frost`), or a path to a model's first `.gguf` shard |
| `--port 8080` | HTTP port |
| `--ctx 262144` | maximum context (256K is the model's native length; memory is only used as needed) |
| `--slots 3` | conversations decoded at once (batched); `1` serves one at a time |
| `--no-mtp` | no speculative decoding |
| anything else | passed to the server (`.venv/bin/python -m serve.server --help`); to keep an option, put it in the model's `server_args` |

| Environment | |
|---|---|
| `BNK_SLOTS=3` | conversations decoded at once |
| `BNK_PARK_GIB=24`, `BNK_SLICE_S=10` | host RAM for parked conversations, and the time slice when requests wait for a slot; `0` turns either off |
| `BNK_READ_SHARE=0.5` | while a prompt is read, the other requests decode for this share of each chunk's time (`0`: reads block them) |
| `BNK_DRAFT_BATCH=0` | draft for each conversation on its own instead of in one batched pass |
| `BNK_LOG_LEVEL=quiet\|info\|debug` | terminal output: `info` prints a line per request and a live status line |
| `BNK_THINK_GUARD=0` | turn the thinking-loop guard off |
| `BNK_DRAFT_VOCAB=` | draft over the whole vocabulary (e.g. for chats in non-Latin scripts) |
| `BNK_MODELS`, `BNK_MTP` | where models and the MTP layer live (`setup.sh` saves them to `bnk.env`) |
| `BNK_API_KEY` | require this key from API clients |
| `BNK_CPU_PIN=1`, `BNK_CPU_CHUNKS=3` | pin the CPU expert workers to cores / split their work dynamically (no measurable gain on this machine, so off by default) |

**Per-model configs** - `configs/<name>.json`:

```json
{
  "sampling": { "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0 },
  "speculation": { "draft": 4, "draft_min_p": 0.8 },
  "thinking_loop_guard": true,
  "server_args": ["--model-id", "orca", "--max-tokens", "32768"],
  "engine_args": ["--threads", "24"]
}
```

This is where a model's persistent settings go; `run.sh <preset>` loads `configs/<preset>.json` (the file named
after the model for a `.gguf` path). `server_args` are any server options (port, context, slots, model id,
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
| Orca: "this model is gated" | accept the terms on its Hugging Face page, `.venv/bin/hf auth login`, run the download again |
| "not enough disk space" | free some space, or put the models on a bigger disk with `--models DIR` |
| A download stopped | run the same command again: finished files are kept, checked files are not re-hashed |
| The server exits at start with an out-of-memory error | not enough RAM for the model's experts (see What you need); close other programs or use IQ3_S |
| `unknown model` from `run.sh` | the preset's files are not where it looks: `BNK_MODELS` in `bnk.env`, or the per-preset variables (`BNK_ABLITERATED`, `BNK_SWIFT`, ...) |
| Port 8080 in use | `./run.sh iq3_s --port 8081` |
| The dashboard is missing (the APIs still work) | run `./setup.sh` again; it rebuilds `serve/web/dist` |

## Repository

```
src/core/      GGUF loading, model config, expert cache, elastic VRAM (vmem)
src/kernels/   CUDA: R-layout GEMV, MoE, QSA sparse attention, DeltaNet, sampling (top-k), GEMM helpers
src/cpu/       expert store, CPU expert pool, AVX2 multi-token kernels
src/engine/    engine (decode, batched decode, prefill, parking), MTP drafter, generator (speculation, sampling)
src/server/    the engine's JSON-lines protocol and request scheduler (bnk serve)
serve/         HTTP server: APIs, chat templates, tokenizer, conversations, telemetry, loop guard, console
serve/web/     dashboard (React + Tailwind + shadcn/ui)
configs/       per-model defaults
docs/          ROADMAP.md: what was measured, what changed, what is next
tools/         CLI, model and MTP downloaders, draft-vocabulary builder, agent benchmarks, batching simulator, checks
tests/         kernel tests and benchmarks
third_party/   ggml (CPU backend: quantization formats, GGUF), MIT
```

## Limitations

* Batched decoding shares at most 8 rows per step between the conversations, so each one verifies fewer drafts
  than alone (3 conversations: ~2.7 rows each).
* A prompt being read shares the GPU with the others' decoding, so it takes longer while others are active.
* Decoding slows as the context fills: the KV cache takes VRAM from the expert cache (up to 7 GiB per conversation
  at 256K).
* The thinking-loop guard catches exact repetition, not paraphrased loops.
* Only sm_70 is built and tested; other GPUs would need their own tuning (and CUDA 13 no longer targets V100).

What is planned next, with measurements behind each item: `docs/ROADMAP.md`.

## Credits

The model architecture and weights are Qwen's; the IQ3_S quantization is ISTA DASLab's, the Uncensored IQ4_XS
orcarouter's, the abliterated transplants SC117's, Swift ukisai's and CYBER-FROST Blackfrost-AI's. bnk vendors
[ggml](https://github.com/ggml-org/ggml) (MIT) for its CPU quantization formats and GGUF parser, and uses
[llama.cpp](https://github.com/ggml-org/llama.cpp) as its numerical reference.
