# bnk

A from-scratch inference engine for **Qwen3.8-Flash-Next-class mixture-of-experts models** (`qwen4exp`) on a
single **NVIDIA V100** with lots of host RAM. The model does not fit in VRAM, so bnk splits it across tiers:
the dense weights and the hottest experts live on the GPU, every expert lives in pinned system RAM, misses are
computed by the CPU during decoding, and experts stream over PCIe while reading prompts. On top of that sit
speculative decoding with the model's own MTP layer, a 256K context whose memory is lent to the expert cache
until it is needed, a thinking-loop guard, an OpenAI/Anthropic-compatible server and a live dashboard.

It is built for, and measured on, this machine:

| | |
|---|---|
| GPU | Tesla V100 32 GB (PG500-216, sm_70), HBM 889 GB/s measured |
| CPU | AMD EPYC 7402, 32 vCPUs (AVX2, no AVX-512) |
| RAM | 128 GB DDR4, ~99 GB/s measured at 24 threads |
| Link | PCIe 3.0 x16, 13.1 GB/s host-to-device measured |
| CUDA | 12.8 (CUDA 13 dropped sm_70) |

## Performance

All numbers from this machine. "Greedy" runs decode deterministically on a warm expert cache; "served" runs go
through the HTTP server with each model's default sampling (configs/), which is what a chat client sees.

**Decoding**

| | IQ3_S (47 GB of experts) | Orca IQ4_XS (61 GB of experts) |
|---|---|---|
| Speculative, greedy, short context | **65 tok/s** | **71 tok/s** |
| Speculative, served (sampled), short context | ~58 tok/s | ~61 tok/s |
| Without speculation, greedy | 53 tok/s | 47 tok/s |
| Speculative at 120K tokens of context | ~45 tok/s | – |
| Speculative at 228K tokens of context | ~38 tok/s | – |

**Reading prompts**

| Prompt | Speed |
|---|---|
| Short prompts (2K-token chunks) | ~525 tok/s |
| 30K tokens (8K-token chunks) | 1,177 tok/s |
| 120K tokens | 1,165 tok/s (1.7 min) |
| 228K tokens | 1,094 tok/s (3.5 min) |

For reference, the engine this was written to replace reached ~57 tok/s decoding and 400–700 tok/s reading
prompts on the same box.

**Correctness.** Logits match llama.cpp's reference implementation token by token (44/44 argmax agreement,
mean KL 0.003 on the validation prompt, the level llama.cpp shows against itself across batch sizes).
Speculative decoding reproduces plain greedy decoding exactly. A needle hidden at 40% depth of a 228K-token
prompt is retrieved verbatim.

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
loads. GEMVs use dp4a with activation sums folding the zero offsets in; small batches use split-K.

**Speculative decoding.** The model's multi-token-prediction layer drafts up to four tokens per step (stopping
early below a confidence cutoff), the main model verifies the window in one pass, and the DeltaNet recurrent
state is replayed for the accepted prefix. Sampled requests use exact rejection sampling, so the output
distribution is the model's own. The drafter scores only a ~145K-token subset of the vocabulary (Latin scripts,
code, symbols, emoji) and attends sparsely through its own indexer, like the main model.

**Long context.** The full-attention layers use the model's QSA sparse attention: a learned indexer scores
compressed 4-token blocks and each query attends to its top 2,048 tokens. bnk computes the block scores as a tiled
fp32 product over 64 queries at a time and runs the attention on tensor cores, one block per KV head serving all
twelve query heads that share it. Prompt reading therefore stays at ~1,100 tok/s from the first token to the
256Kth.

**Elastic VRAM.** The KV cache, the indexer keys, the drafter's KV and the prompt-path buffers each reserve
address space for their maximum and map physical memory in 2 MiB pages only as needed. All of them draw on one
budget whose remainder is the expert cache. When a conversation grows or a long prompt arrives, the cache gives
up its coldest slots (moving hot experts out of the freed pages first); afterwards it grows back. A 256K context
therefore costs nothing until it is used: short chats keep a 20.7 GiB expert cache, the same as with a small
context.

**Reading prompts.** Prompts run in chunks (2K tokens, 8K for long prompts) with tensor-core GEMMs: resident
experts run grouped GEMMs, non-resident ones stream over PCIe one layer ahead into double-buffered staging, and
the CPU takes the experts that are cheaper for it.

**Thinking-loop guard.** Reasoning models occasionally loop. The server watches the thinking tokens and, when
most of a recent 4K-token window repeats earlier 12-token stretches, lets the model finish its sentence, closes
the reasoning with a short note, and lets it answer in the budget left. The hand-over reuses the engine's state,
so it costs ~0.3 s however long the conversation. Real reasoning traces stay under 5% repetition against a 55%
trigger.

## Supported models

Any GGUF of the `qwen4exp` architecture (Qwen3.8-Flash-Next and fine-tunes): 48 layers (36 gated DeltaNet + 12
QSA full-attention), 512 routed experts (top 10) plus a shared expert, hyper-connections (4 streams), the
per-layer n-gram embedding. Expert formats: IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q2_0, Q4_K, Q5_K, Q6_K, Q8_0.
Tested with:

* **IQ3_S** — `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (ISTA DASLab), 2 shards
* **Orca IQ4_XS** — `Qwen3.8-Flash-Next-Uncensored-IQ4_XS`, 3 shards

Other quantizations of the same architecture (for example Unsloth's UD-IQ4_XS) use the same formats and should
load as they are; give them a `configs/<name>.json` to set their sampling.

## Setup

**Requirements:** Linux, an sm_70 GPU (V100) with CUDA 12.8, CMake ≥ 3.24, g++ ≥ 12 (C++20), Python ≥ 3.10,
Ninja recommended, enough RAM to page-lock every expert (47 GB for IQ3_S, 61 GB for Orca) plus ~10 GB.

```bash
git clone https://github.com/christ-pher/bnk.git && cd bnk

# 1. the engine
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. the server's Python environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. the MTP draft layer, built from the official Qwen checkpoint (downloads only the mtp.* tensors, ~5 GB)
.venv/bin/python tools/build_mtp.py --out /opt/models/bnk/mtp/mtp-q2_0.gguf

# 4. run (builds the dashboard and the per-model draft vocabulary on first start)
./run.sh iq3_s            # or: ./run.sh orca, or ./run.sh /path/to/model-00001-of-0000N.gguf
```

Then open `http://<host>:8080` for the dashboard. `run.sh` looks for models under `$BNK_MODELS`
(default `/opt/models/Strata/models`); edit the presets at the top of `run.sh` for other locations.

First start takes about a minute: the experts are copied into page-locked RAM and the VRAM cache is filled.
The cache ranking learned while serving is saved to `~/.cache/bnk/counts-<model>.bnkc` and used next time.

## Using it

**run.sh**

| Option | |
|---|---|
| `--port 8080` | HTTP port |
| `--ctx 262144` | maximum context (256K is the model's native length; memory is only used as needed) |
| `--no-mtp` | no speculative decoding |
| anything else | passed to the server (see `python -m serve.server --help`) |

| Environment | |
|---|---|
| `BNK_LOG_LEVEL=quiet\|info\|debug` | terminal output: `info` prints a line per request and a live status line |
| `BNK_THINK_GUARD=0` | turn the thinking-loop guard off |
| `BNK_DRAFT_VOCAB=` | draft over the whole vocabulary (e.g. for chats in non-Latin scripts) |
| `BNK_MODELS`, `BNK_MTP` | where models and the MTP layer live |

**Per-model configs** — `configs/<name>.json`:

```json
{
  "sampling": { "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0 },
  "speculation": { "draft": 4, "draft_min_p": 0.8 },
  "thinking_loop_guard": true
}
```

Sampling values are defaults: a request's own fields win. The shipped speculation settings come from sweeps on
this machine (draft 4 with a 0.8 confidence cutoff beat the engine's 3 / 0.5 by 13–18% for both models).

**APIs** — OpenAI `POST /v1/chat/completions` and `/v1/completions` (streaming, tools, `reasoning_content`),
Anthropic `POST /v1/messages`, `GET /v1/models`. Thinking is on by default; turn it off per request with
`"chat_template_kwargs": {"enable_thinking": false}`. Requests may also set `draft` and `draft_min_p`.
Set `BNK_API_KEY` to require a key.

```bash
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

**Dashboard** — live analytics built with [shadcn/ui](https://ui.shadcn.com/): decode and prompt speeds,
speculation, expert-cache hit rates per layer, GPU / PCIe / CPU / memory, requests, the engine log, and a chat
panel for testing. Data arrives over server-sent events (`GET /api/stream`); `GET /api/stats` returns a snapshot.

**Engine CLI** — `build/bnk` also runs on its own:

```bash
build/bnk run   --model M --tokens-file ids.csv --max-new 200 --mtp mtp.gguf   # generate, with timings
build/bnk check --model M --tokens-file ids.csv --ref ref/prefix              # vs llama.cpp, layer by layer
build/bnk pdump --model M --tokens-file ids.csv --ref out.bin                 # prompt pass, write logits
```

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
tools/         CLI, MTP builder, draft-vocabulary builder, llama.cpp reference dumper
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
[ggml](https://github.com/ggml-org/ggml) (MIT) for its CPU quantization formats and GGUF parser, and used
[llama.cpp](https://github.com/ggml-org/llama.cpp) as the numerical reference throughout.
