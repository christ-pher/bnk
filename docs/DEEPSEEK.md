# DeepSeek-V4.1-Flash-1017 in bnk (branch `deepseek`, 2026-10-23)

Goal: run `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` (GGUF architecture `deepseek41`) on the two V100s
and 192 GB of RAM: a 763B-parameter model whose weights do not fit in memory at any useful quantization unless
the two 98 GB Engram tables stay on disk and the routed experts are split between VRAM and host RAM without a copy.

## The model

| | DeepSeek-V4.1-Flash | Qwen3.8-Flash-Next (bnk's MoE engine) |
|---|---|---|
| Layers | 40, every one attention + MoE | 48: 36 DeltaNet + 12 attention |
| Width | 5,120 | 2,560 |
| Residual | 4 hyper-connection streams, mixes from the stream itself, `comb` made doubly stochastic by 20 Sinkhorn iterations | 4 streams, low-rank down/up mixer |
| Attention | 64 heads x 512, **one** shared KV head (the latent is both K and V), low-rank Q (1,280) and grouped low-rank output (8 groups x 1,024), RoPE on 64 dims (adjacent pairs), per-head sinks | 24 heads / 2 KV heads x 256, q|gate per head, NeoX RoPE |
| Context | a sliding window of 128 raw latents per layer, plus (38 of 40 layers) compressed latents shared from the nearest source layer (2, 8, 14, 20; ratio 2 pools pairs with a learned softmax gate, ratio 1 is a projection), each query keeping the 512 scored best by an indexer at 8 layers (candidate blocks of 8 at layer 20 prefilter later ones to 2,048 blocks); YaRN x16 on the compressed path | QSA: top-k blocks of 4 cells |
| MoE | 384 experts, top-6 of sqrt(softplus(logits)) + bias, renormalized, x1.5; one shared expert; SwiGLU with up clamped to [-10, 10] and gate to <= 10 | 512, top-10 softmax, shared expert with its own gate |
| Engram | n-gram hash tables at layers 1 and 14: 24 rows of 256 per token (2-, 3-, 4-grams x 8 heads over a case/space-folded token map), a 6,144 -> 25,600 projection into a key per stream and one value, a signed-sqrt sigmoid gate on the key/stream dot | PLE: one layer, 16 rows, conv |
| Drafter | DSpark (3 MoE layers, separate GGUF) - not used yet | MTP layer |
| Vocabulary | 129,280, GPT-2 BPE with the `joyai-llm` pre-tokenizer; chat template in the GGUF (`<think>`, DSML tool calls) | 248,320 |

The GGUF (ISTA's variant of llama.cpp PR 30817's layout): routed experts as `blk.N.ffn_{gate,up,down}_exps.weight`
in the RCO-chosen formats (IQ1_M, IQ2_XXS, IQ2_XS, ... per tensor), dense weights Q5_K / Q8_0, routers,
compressor, indexer and Engram gate weights BF16, norms / hyper-connection mixers / sinks F32; the Engram hashing
constants as tensors (`engram_token_map`, `engram_primes` [layer][24], `engram_offsets` relative to each layer's
table, `engram_multipliers` [layer][4]); the tables in a separate file (`deepseek41.engram.file`,
architecture `deepseek41-engram`, `blk.{1,14}.engram_embd.weight` IQ4_NL [256 x 384M]).

## Memory plan

* **Engram tables** (110.6 GB): memory-mapped from the NVMe, read 48 rows (144 bytes each) per token on the host,
  dequantized on the GPU. Never resident.
* **Dense weights** (~6 GB, Q5_K / Q8_0 / F16 / F32): VRAM, split over the two GPUs as a layer pipeline
  (default split after layer 19: the compressed-KV source layer 20 and its consumers stay on one card).
* **Routed experts** (the model file less ~6 GB): *disjoint placement*. The ranking (routing counts from earlier
  runs, else uniform) and each GPU's free VRAM decide which experts are resident in the VRAM cache; only the
  others are copied into the pinned host arena (`ExpertStore::build` with a skip list; `ExpertCache` fills
  resident experts straight from the file). The cache is static (no adaptive swaps: a swapped-in expert would
  have no host copy). Misses run on the CPU pool as in the Qwen engine. Budget: ~48 GB of VRAM + ~180 GB of RAM,
  so a model file of up to ~210 GB fits (the "largest quant that fits" rule: IQ3_XXS if its total is at or
  under that, else IQ2_XS).
* **KV**: a ring of 136 (window 128 + the window's rows) latents of 512 fp16 per layer; compressed latents (512)
  and indexer keys (128) per source layer, fp16, mapped as the context grows (ratio 2: half a row per token).
  Under 1 GB per conversation at 256K tokens.

## Where the code lives

`src/deepseek/`: `ds_model` (config, weight upload, the expert "view" for the shared expert machinery),
`ds_ops.cu` (hyper-connection mixes + Sinkhorn, pre/post, pair RoPE with YaRN, window + compressed-KV attention
with sinks, compressor pooling, indexer scoring / candidate blocks / top-k by radix select, Engram gate,
sqrt-softplus routing), `ds_engine` (decode windows as CUDA graphs, slots, commit of the compressor's partial
group, the pipeline over two GPUs, prompts through decode windows for now). Shared code changed:
`ExpertStore`/`ExpertCache` (skip list, file fill), the MoE activation and the CPU pool (SwiGLU clamp), `bnk`
(`general.architecture == deepseek41` picks `DsEngine`), `serve/tokenizer.py` (`joyai-llm`), `run.sh` (the
Engram file as `--engram`).

## Status

See the bottom of this file as the work progresses.

* 2026-10-23: design; loader, kernels (`tests/test_ds_ops.cu` passes against CPU references at the model's shapes),
  engine, configs and the `joyai-llm` tokenizer (exact against DeepSeek's tokenizer.json) written; builds clean.
  ISTA published three builds (IQ1_M 121 GB, IQ2_XXS 138 GB, IQ2_XS 155 GB) and recommends IQ2_XXS; IQ2_XS and
  IQ2_XXS are being downloaded for a side-by-side. No llama.cpp runs these files yet (the Engram layout differs from
  PR 30817), so correctness is judged by generation quality and NLL for now. Prompts still go through decode
  windows; no drafter.
* 2026-10-24: first run generated correct text; decode (1-token windows) then degraded to loops: the engine never
  called `ggml_cpu_init()`, so ggml's single-token `vec_dot` kernels (the CPU pool's path for an expert with one
  token) read an unfilled fp16 table and returned zeros (ppl 17 at 1-token windows vs 3.2 at 8). One line. Tests
  added: `tests/test_ds_gemv.cu` (every GEMV path at 1-8 tokens), `tests/test_ds_cpu.cpp` (the CPU pool on real
  experts, with a mailbox replay mode); `bnk hdump` takes `BNK_HDUMP_T` / `BNK_HDUMP_T2`, the engine `BNK_DS_TRACE`
  / `BNK_DS_TRACE_DUMP`. The model writes its tool calls as `<｜DSML｜ calls>` / `<｜DSML｜ invoke ...>` (a space
  and a short word after the one-token marker, not the template's `tool_calls`); the server's parser accepts both
  spellings (Strata 0.2.4, which added DeepSeek the same night, documents the same). Served: a reasoning question
  answered correctly at 13.8 tok/s decode; tool calls parsed.
* 2026-10-24 (later): the batched prompt path (`src/deepseek/ds_prefill.cpp`): chunks of up to 2,048 tokens through
  fp16 tensor-core GEMMs, the chunk kernels, and the routed experts on the tensor cores (resident from the cache, the
  most-used others streamed into a staging arena of 1 GiB per buffer, `BNK_DS_STAGE_MIB`, the rest on the CPU). The
  window keys of a chunk go through a linear buffer (the 136-slot ring cannot hold a chunk) and the ring is refilled
  after it. Validated: the copy test with its prefix read through the chunked path scores ppl 1.002 (every token right);
  671 tokens in 5.7 s (118 tok/s) against ~20 through decode windows; the path is
  PCIe-bound on expert streaming (every layer's experts are used by a long chunk), so prefetching the next layer's
  experts during the current layer and overlapping the two GPUs' chunks are the next steps.
