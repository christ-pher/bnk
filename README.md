# bnk

A from-scratch inference engine for **Qwen3.8-Flash-Next-class mixture-of-experts models** (`qwen4exp`) on a
single **NVIDIA V100** with lots of host RAM. The model does not fit in VRAM, so bnk splits it across tiers:
the dense weights and the hottest experts live on the GPU, every expert lives in pinned system RAM, misses are
computed by the CPU during decoding, and experts stream over PCIe during prompt processing. On top of that, bnk
provides speculative decoding with the model's own MTP layer, a 256K context whose memory is lent to the expert
cache until it is needed, a thinking-loop guard, an OpenAI/Anthropic-compatible server and a live dashboard.

It is built for, and measured on, this machine:

| Component | Specification |
|---|---|
| GPU | Tesla V100 32 GB (PG500-216, sm_70), HBM 889 GB/s measured |
| CPU | AMD EPYC 7402, 32 vCPUs (AVX2, no AVX-512) |
| RAM | 128 GB DDR4, ~99 GB/s measured at 24 threads |
| Link | PCIe 3.0 x16, 13.1 GB/s host-to-device measured |
| CUDA | 12.8 (CUDA 13 dropped sm_70) |

## Performance

All numbers were measured on this machine. "Greedy" runs decode deterministically with a warm expert cache;
"served" runs go through the HTTP server with each model's default sampling (`configs/`), which is what a chat
client sees.

**Decoding**

| | IQ3_S (47 GB of experts) | Orca IQ4_XS (61 GB of experts) |
|---|---|---|
| Speculative, greedy, short context | **65 tok/s** | **71 tok/s** |
| Speculative, served (sampled), short context | ~58 tok/s | ~61 tok/s |
| Without speculation, greedy | 53 tok/s | 47 tok/s |
| Speculative at 120K tokens of context | 44 tok/s | 59 tok/s |
| Speculative at 228K tokens of context | 43 tok/s | 52 tok/s |

The long-context rows decode 300 tokens after a prompt of real source code (llama.cpp's), greedily, with the
models' speculation settings.

**Prompt processing** — the same source-code prompts, read from scratch:

| Prompt length | IQ3_S | Orca IQ4_XS |
|---|---|---|
| ~1.8K tokens | 338 tok/s | 290 tok/s |
| ~30K tokens | 685 tok/s | 628 tok/s |
| ~120K tokens | 714 tok/s (2.8 min) | 666 tok/s (3.0 min) |
| ~228K tokens | 922 tok/s (4.1 min) | 848 tok/s (4.5 min) |

Code routes to a wide spread of experts, so most of a chunk's experts stream over PCIe; text with narrower
routing processes faster (a synthetic 228K-token prompt ran at 1,094 tok/s).

**Long conversations** — the workload that matters most in practice. Logs of real use show agent-style sessions:
a median context of 56–66K tokens (a quarter of requests above 100K), ~97% of each prompt shared with the previous
turn, and five to six times more time spent generating than processing prompts. Measured on an 80K-token coding
conversation with Orca (`tools/agent_bench.py`: the repository's own sources as context, then turns that each
add a tool result and a question):

| | Orca IQ4_XS at ~80K tokens of context |
|---|---|
| Decode, greedy | 65–74 tok/s |
| Decode, served (temperature 0.6) | 58 tok/s |
| Expert misses (served by the CPU) | 7–9% |
| Starting a follow-up turn | ~0.5 s for a short message, ~4 s for a 1K-token tool result |

Runs of this benchmark vary by about ±5% between batches (the adaptive expert cache, GPU clocks, the VM's host),
so comparisons are made back to back and repeated.

**CYBER-FROST-3.8 (Q5_K_M)** — measured back to back with Orca on the same prompts (`bnk run`, greedy, warm
routing counts, each model's speculation settings; the short-context rows average three chat prompts, the others
decode 300 tokens after llama.cpp source code):

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
| 80K agent conversation, served (temperature 1.0) | 43 tok/s, 4.7–5.6 s per turn read | |

Its experts are 28% larger than Orca's, so fewer fit in VRAM (27% of them vs 34%): more of each token's experts
run on the CPU, which is bound by RAM bandwidth, and decoding is 10–22% slower (more so at long context). Speculation at temperature 1.0
gains most from a high draft cutoff (draft 5 / cutoff 0.8: ~47.5 tok/s served at short context, against ~41 with
the engine defaults 3 / 0.5).

For reference, the engine bnk replaces reached ~57 tok/s for decoding and 400–700 tok/s for prompt processing
on the same machine.

**Correctness.** Logits match llama.cpp's reference implementation token by token: 44/44 argmax agreement and a
mean KL divergence of 0.003 on the validation prompt, comparable to the variation llama.cpp shows against itself
across batch sizes. Speculative decoding reproduces plain greedy decoding exactly, and a needle hidden at 40%
depth in a 228K-token prompt is retrieved verbatim.

## How it works

```
                 ┌────────────────────── V100 32 GB ───────────────────────┐
  tokens ──────► │ dense weights (attention, DeltaNet, hyper-connections)  │
                 │ VRAM expert cache: hottest experts, adaptive  ◄─┐ elastic budget
                 │ KV + indexer keys: mapped as the context grows ◄┤ (CUDA VMM)
                 │ prompt buffers: borrowed for long prompts      ◄┘
                 └──────┬───────────────────────────────▲──────────────────┘
               misses   │ mapped-memory mailbox          │ PCIe streaming (prompts)
                        ▼                                │
                 ┌─────────── host: 128 GB RAM, 24 threads ───────────────┐
                 │ every expert, page-locked · AVX2 expert kernels         │
                 └─────────────────────────────────────────────────────────┘
```

**Decoding.** Each step runs the whole layer stack as one captured CUDA graph. The router's choices are
deduplicated on the GPU; experts resident in VRAM run there, and the misses for the layer are posted to a
host-mapped mailbox that a pool of 24 CPU threads answers while the GPU carries on. The CPU kernels are bnk's own
AVX2 code (bit-identical to ggml's dot products) that decode each weight row once for all the tokens of a
speculative window. The expert cache starts from the routing counts learned in earlier sessions and keeps
adapting: every few steps it swaps the hottest non-resident experts in, asynchronously, on a copy stream.

**Weights on the GPU.** Quantized matrices (Q4_K–Q8_0, IQ2_S–IQ4_XS, Q2_0, BF16) are repacked at load into an
"R layout": a lossless struct-of-arrays rearrangement of each row's blocks so every warp issues coalesced 16-byte
loads. GEMVs use dp4a, with activation sums that fold in the zero offsets; small batches use split-K.

**Speculative decoding.** The model's multi-token-prediction layer drafts up to four tokens per step, stopping
early when its confidence drops below a cutoff. The main model then verifies the whole window in one pass, and
the DeltaNet recurrent state is replayed for the accepted prefix. Sampled requests use exact rejection sampling,
so the output distribution is the model's own. The drafter scores only a ~145K-token subset of the vocabulary
(Latin scripts, code, symbols, emoji) and, like the main model, attends sparsely through its own indexer.

**Long context.** The full-attention layers use the model's QSA sparse attention: a learned indexer scores
compressed 4-token blocks and each query attends to its top 2,048 tokens. bnk computes the block scores as a tiled
fp32 product over 64 queries at a time and runs the attention on tensor cores, one block per KV head serving all
twelve query heads that share it, so prompt processing does not slow down as the context fills: it is bound by
streaming experts over PCIe, not by attention, from the first token to the 256Kth.

**Elastic VRAM.** The KV cache, the indexer keys, the drafter's KV and the prompt-processing buffers each reserve
address space for their maximum and map physical memory in 2 MiB pages only as needed. All of them draw on one
budget whose remainder is the expert cache. When a conversation grows or a long prompt arrives, the cache gives
up its coldest slots (moving hot experts out of the freed pages first); afterwards it grows back. A 256K context
therefore costs nothing until it is used: short chats keep a 20.7 GiB expert cache, the same as with a small
context.

**Prompt processing.** Prompts are processed in chunks (2K tokens, or 8K for long prompts) with tensor-core
GEMMs: resident experts run as grouped GEMMs, non-resident ones stream over PCIe one layer ahead into
double-buffered staging, and the CPU takes the experts that are cheaper to compute there.

**Conversations.** Chat clients re-send the whole conversation each turn and usually re-render earlier turns
(Qwen's template drops a previous answer's reasoning), so a new prompt shares only part of what the engine has
already processed. The KV cache is stored per position and stays valid, but the DeltaNet layers carry running
state, so bnk snapshots that state (~75 MB, kept in pinned RAM) where each prompt's last turn begins. The next
request rewinds to the latest snapshot inside the shared prefix and processes only what is new: a follow-up turn
in an 80K-token conversation starts in under a second instead of re-processing the whole history.

**Thinking-loop guard.** Reasoning models occasionally loop. The server watches the thinking tokens and, when
most of a recent 4K-token window repeats earlier 12-token stretches, lets the model finish its sentence, closes
the reasoning with a short note, and lets the model answer within the remaining token budget. The hand-over
reuses the engine's state, so it costs ~0.3 s regardless of the conversation's length. Real reasoning traces
stay under 5% repetition, well below the 55% trigger.

## Supported models

Any GGUF of the `qwen4exp` architecture (Qwen3.8-Flash-Next and fine-tunes): 48 layers (36 gated DeltaNet + 12
QSA full-attention), 512 routed experts (top 10) plus a shared expert, hyper-connections (4 streams), the
per-layer n-gram embedding. Expert formats: IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q2_0, Q4_K, Q5_K, Q6_K, Q8_0. Files that
also carry an MTP block (`nextn_predict_layers`) load their trunk.
Tested with:

* **IQ3_S** — `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (ISTA DASLab), 2 shards
* **Orca IQ4_XS** — `Qwen3.8-Flash-Next-Uncensored-IQ4_XS`, 3 shards
* **CYBER-FROST-3.8** — Blackfrost-AI's fine-tune, as `peasantsmith/CYBER-FROST-3.8-PS-GUFF` (Q5_K_M: Q5_K
  gate/up and IQ4_NL down experts with an importance matrix, 78 GB of experts), stored without its PLE table
  (see below)

**CYBER-FROST-3.8.** Of the published quantizations of this fine-tune, PS-GUFF has the most precise experts that fit
this machine's RAM and needs no new kernels (the other GGUFs use Q5_0/Q5_1/Q3_K/MXFP4 experts, or need more than
the 128 GB of RAM). Three things about the file:

* Its 28.8 GB PLE table is byte-identical to Orca's (checked with a SHA-256 of the whole tensor), so it is
  downloaded without it (`tools/fetch_gguf.py`, 83.7 GB instead of 112.5 GB) and the engine reads the table from
  the Orca file (`--ple-gguf`; `run.sh cyber-frost` passes it).
* It declares its MTP block (`block_count` 49, `nextn_predict_layers` 1); the engine runs the 48-layer trunk and
  drafts with `--mtp` as usual. That MTP head is byte-identical to the official Qwen one, so the stock draft
  layer from `tools/build_mtp.py` is the model's own.
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

**GSQ-RCO-abliterated IQ3_S** ([SC117](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)): the
IQ3_S above with 144 residual-writing tensors swapped for Orca's abliterated ones (mostly Q8_0 now); shard 2 and the
MTP head are the stock ones. It starts from Orca's settings (temperature 0.6, guard on); not tuned yet.

```bash
hf download SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF --include "IQ3_S/*" --local-dir /opt/models/gsq-rco-abliterated
./run.sh abliterated      # BNK_ABLITERATED overrides the path
```

Other quantizations of the same architecture (for example Unsloth's UD-IQ4_XS) use the same formats and should
load as-is; add a `configs/<name>.json` to set their sampling defaults.

## Setup

**Requirements:** Linux, an sm_70 GPU (V100) with CUDA 12.8, CMake ≥ 3.24, g++ ≥ 12 (C++20), Python ≥ 3.10,
Ninja (recommended), and enough RAM to page-lock every expert (47 GB for IQ3_S, 61 GB for Orca, 78 GB for
CYBER-FROST) plus ~10 GB.

```bash
git clone https://github.com/christ-pher/bnk.git && cd bnk
./setup.sh --mtp --models /path/to/your/models
./run.sh iq3_s            # or: ./run.sh orca, ./run.sh cyber-frost, or ./run.sh /path/to/model-00001-of-0000N.gguf
```

`setup.sh` does everything in one pass:

1. It checks the GPU, the driver, CUDA 12.x, RAM and AVX2.
2. It installs missing system packages with `apt` (asking first; `--yes` skips the question, `--no-system` never
   touches them) and uses pip-provided CMake and Ninja when the system's are too old.
3. It builds the engine, the Python environment and the dashboard.
4. With `--mtp`, it builds the MTP draft layer from the official Qwen checkpoint (downloading only the `mtp.*`
   tensors, ~5 GB).
5. With `--models`, it saves your model directory to `bnk.env`, which `run.sh` reads.

The CUDA toolkit itself has to come from NVIDIA (`cuda-toolkit-12-8` from their apt repository); `setup.sh`
points to it when it is missing. To do the same steps by hand:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/build_mtp.py --out /opt/models/bnk/mtp/mtp-q2_0.gguf
```

Then open `http://<host>:8080` for the dashboard. `run.sh` looks for models under `$BNK_MODELS` (set by
`setup.sh --models`); the `iq3_s` and `orca` presets at the top of `run.sh` name the files it expects there (`cyber-frost` looks
in `/opt/models/cyber-frost` and takes the PLE table from the Orca file).

First start takes about a minute: the experts are copied into page-locked RAM and the VRAM cache is filled.
The cache ranking learned while serving is saved to `~/.cache/bnk/counts-<model>.bnkc` and used next time.

## Using it

**run.sh**

| Option | |
|---|---|
| `--port 8080` | HTTP port |
| `--ctx 262144` | maximum context (256K is the model's native length; memory is only used as needed) |
| `--no-mtp` | no speculative decoding |
| anything else | passed to the server (see `python -m serve.server --help`); to keep an option, put it in the model's `server_args` |

| Environment | |
|---|---|
| `BNK_LOG_LEVEL=quiet\|info\|debug` | terminal output: `info` prints a line per request and a live status line |
| `BNK_THINK_GUARD=0` | turn the thinking-loop guard off |
| `BNK_DRAFT_VOCAB=` | draft over the whole vocabulary (e.g. for chats in non-Latin scripts) |
| `BNK_MODELS`, `BNK_MTP` | where models and the MTP layer live |
| `BNK_CPU_PIN=1`, `BNK_CPU_CHUNKS=3` | pin the CPU expert workers to cores / split their work dynamically (Strata's scheme; no measurable gain on this machine, so off by default) |

**Per-model configs** — `configs/<name>.json`:

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
after the model for a `.gguf` path). `server_args` are any server options (`python -m serve.server --help`: port,
context, model id, completion budget, API key ...) and are read as if typed before the command line, so options
given to `run.sh` still win. `engine_args` go to the engine (`--threads`, `--cache-gib`, ...; see
`tools/bnk_main.cpp`). Sampling values are defaults; a request's own fields take precedence. The shipped speculation settings come from
sweeps on this machine: drafting up to 4 tokens with a 0.8 confidence cutoff was 13–18% faster than the engine's
defaults (3 tokens, 0.5 cutoff) for both models.

**APIs** — OpenAI `POST /v1/chat/completions` and `/v1/completions` (streaming, tools, `reasoning_content`),
Anthropic `POST /v1/messages`, `GET /v1/models`. Thinking is on by default; turn it off per request with
`"chat_template_kwargs": {"enable_thinking": false}`. Requests may also set `draft` and `draft_min_p`.
Set `BNK_API_KEY` to require a key.

```bash
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

**Dashboard** — live analytics built with [shadcn/ui](https://ui.shadcn.com/): decode and prompt-processing speeds,
speculation, expert-cache hit rates per layer, GPU / PCIe / CPU / memory, requests, the engine log, and a chat
panel for testing. The sidebar lists the OpenAI and Claude API base URLs and the served model id, each with a copy
button. Data arrives over server-sent events (`GET /api/stream`); `GET /api/stats` returns a snapshot.

**Engine CLI** — `build/bnk` also runs on its own:

```bash
build/bnk run   --model M --tokens-file ids.csv --max-new 200 --mtp mtp.gguf   # generate, with timings
build/bnk check --model M --tokens-file ids.csv --ref ref/prefix              # vs llama.cpp, layer by layer
build/bnk pdump --model M --tokens-file ids.csv --ref out.bin                 # process a prompt, write its logits
```

`BNK_SPLIT=N` makes `pdump` time only the tokens after the first N (a follow-up turn at depth), and
`BNK_PROFILE_DECODE=1` limits an `nsys --capture-range=cudaProfilerApi` profile of `bnk run` to decoding.

`tools/llama_ref.cpp` dumps llama.cpp's logits and per-layer states for `check`.

## Repository

```
src/core/      GGUF loading, model config, expert cache, elastic VRAM (vmem)
src/kernels/   CUDA: R-layout GEMV, MoE, QSA sparse attention, DeltaNet, GEMM helpers
src/cpu/       expert store, CPU expert pool, AVX2 multi-token kernels
src/engine/    engine (decode/prefill), MTP drafter, generator (speculation, sampling)
src/server/    the engine's JSON-lines protocol (bnk serve)
serve/         HTTP server: APIs, chat templates, tokenizer, telemetry, loop guard, console
serve/web/     dashboard (React + Tailwind + shadcn/ui)
configs/       per-model defaults
tools/         CLI, MTP builder, draft-vocabulary builder, GGUF fetcher, agent benchmark, llama.cpp reference dumper
tests/         kernel tests and benchmarks
third_party/   ggml (CPU backend: quantization formats, GGUF), MIT
```

## Limitations

* One request at a time (requests queue); the engine is tuned for a single interactive user.
* Decoding slows as the context fills: the KV cache takes VRAM from the expert cache (up to 7 GiB at 256K).
* The thinking-loop guard catches exact repetition, not paraphrased loops.
* Only sm_70 is built and tested; other GPUs would need their own tuning (and CUDA 13 no longer targets V100).

## Credits

The model architecture and weights are Qwen's; the IQ3_S quantization is ISTA DASLab's. bnk vendors
[ggml](https://github.com/ggml-org/ggml) (MIT) for its CPU quantization formats and GGUF parser, and uses
[llama.cpp](https://github.com/ggml-org/llama.cpp) as its numerical reference.
