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
| Drafter | DSpark, the model's own draft head: 3 MoE layers of 128 experts (top-3), blocks of 5 drafts, Markov and confidence heads (`docs/DSPARK.md`; `src/deepseek/ds_spark.cpp`): code 30.8 -> 45-52 tok/s, 89% of its drafts accepted; plus the prompt-lookup drafts | MTP layer |
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
  resident experts straight from the file). While serving, the cache follows the routing by *exchange*
  (`ExpertCache::adapt_exchange`, every `BNK_DS_ADAPT` = 4 decode forwards, up to `BNK_DS_ADAPT_SWAPS` = 8 in flight a
  GPU, `BNK_DS_ADAPT=0` static): a layer's hottest non-resident expert takes the slot of its coldest resident one,
  which first gets a host copy in one of the store's two spare blobs a layer (a helper thread writes it from the
  mapped model file), then the slot is unmapped, refilled from the incoming expert's host blob and mapped, and the
  incoming expert's blob returns to the store as the spare. Every expert is resident or in the store at every moment;
  every forward waits (on the GPU) for the exchange's copies in flight before it starts.
  The routing-counts file that ranks the next start is capped (`BNK_COUNTS_CAP`, 2^28 routings; halved past it),
  so it reflects the recent workload rather than everything ever served. Misses run on the CPU pool as in the Qwen engine. Budget: ~48 GB of VRAM + ~180 GB of RAM,
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
  answered correctly at 13.8 tok/s decode; tool calls parsed. (2026-10-25: the spelling difference was the GGUF's
  template being wrong, not the model: see below.)
* 2026-10-25, agent session review (branch `ds-agent-fixes`): a coding agent at 85K tokens of context reported
  "malformed tokens" in the model's output. Findings, with the engine's output text unavailable (the server does
  not log it): the sampling was the model card's (temperature 1.0, top_p 1.0; DeepSeek's agentic evaluations used
  top_p 0.95, DeepSeek Harness itself sets no sampling fields), the tokenizer and stream decoder round-trip
  byte-exactly, string tool arguments pass verbatim, RoPE/YaRN match llama.cpp's DeepSeek-V4 graph (window layers
  plain, compressed layers YaRN x16 from 64K, no attention factor). Three defects in this stack, all fixed here:
  * **The prompt spoke V4's tool-call dialect.** DeepSeek's `encoding/encoding.py` (the prompt-format reference for
    V4.1) writes `<｜DSML｜ calls>`, `<｜DSML｜ invoke`, `<｜DSML｜ parameter`; its README: "the V4 format used
    `<｜DSML｜tool_calls>` without a space". The template embedded in the GSQ-RCO GGUFs (and llama.cpp's shipped
    DeepSeek-V4 templates) is V4's, so the tools header taught the model the wrong spelling and every earlier call
    of its own was re-rendered in it (`tool` `_c` `alls` vs ` calls` at the token level), 20+ turns deep. The
    server now patches the template at load for a V4.1 model (`serve/chat.py` `dsml_v41_template`; the startup log
    says so); `tests/test_tool_calls.py` renders a tool-call conversation through the real GGUF's template and
    checks it against the encoder's output.
  * **A tool call cut off at max_tokens came back as raw markup in `content`.** The parser only extracts complete
    blocks; the client got `<｜DSML｜ invoke ...` as text (finish `length`). Now the content before the call is
    returned, the markup dropped, `timings.truncated_tool_call` set, and the console says `✂ tool call cut off`.
  * **Verbatim repetition in the answer was unguarded.** The loop guard watched reasoning only; the model repeated
    a tool call for 27,700 of 32,000 tokens (100% lookup-draft acceptance) until the cap. A second guard watches
    the answer (`content_guard()`: from 2,048 tokens, 65% of 12-token spans repeated within 4K; `BNK_CONTENT_GUARD=0`,
    `--no-content-guard`, config `content_loop_guard`), ends the request (`timings.content_loop_guard`): complete
    calls parsed so far are kept once each (identical repeats collapse), the call being repeated when it was cut off
    is dropped as above, and without calls the finish is `length`. `tests/test_loop_guard.py`, and
    `tests/test_serve_run.py` drives the server's request loop against a fake engine through all three cases.
  Not verifiable without the GPUs: output quality past 64K tokens of context, and the lookup-draft rollback path
  under rejections.
* 2026-10-24 (later): the batched prompt path (`src/deepseek/ds_prefill.cpp`): chunks of up to 2,048 tokens through
  fp16 tensor-core GEMMs, the chunk kernels, and the routed experts on the tensor cores (resident from the cache, the
  most-used others streamed into a staging arena of 1 GiB per buffer, `BNK_DS_STAGE_MIB`, the rest on the CPU). The
  window keys of a chunk go through a linear buffer (the 136-slot ring cannot hold a chunk) and the ring is refilled
  after it. Validated: the copy test with its prefix read through the chunked path scores ppl 1.002 (every token right);
  671 tokens in 5.7 s (118 tok/s) against ~20 through decode windows; the path is
  PCIe-bound on expert streaming (every layer's experts are used by a long chunk), so prefetching the next layer's
  experts during the current layer and overlapping the two GPUs' chunks are the next steps.
* 2026-10-24 (the first performance pass, branch `ds-perf`; the state and the plan are in `docs/DEEPSEEK-NEXT.md`): the
  prompt path became layer-major (a group of chunks per layer, so each non-resident expert crosses PCIe once per layer per
  group), streams experts in waves through the two staging buffers with the next layer's experts prefetched from its
  router applied one layer early, keeps the compute stream's small transfers off the copy engine (they queued behind the
  prefetch), and converts strided GEMM inputs in one launch (not one per row). Prompts: 2,048 tokens 111 -> 279 tok/s,
  8,192 tokens ~110 -> 467 tok/s end to end (both stages in sequence for one group). Decode: the CPU pool claims work
  dynamically (`BNK_CPU_CHUNKS` 8), reads Engram rows on several threads, and prompt-lookup drafting runs (copying text:
  12.3 -> 32 tok/s; plain decode 12.95 -> 17.0 tok/s with the split-key attention and the multi-block mixes for small
  windows). Multi-row CPU kernels, batched-key attention and SMT threads were measured slower and dropped.
* 2026-10-26, later (`docs/DSPARK.md`): the DSpark draft head ported (its GGUF through upstream llama.cpp's `--dspark`
  converter, experts MXFP4 in RAM on the CPU pool): code decode 30.8 -> 45 tok/s (52 with the cache warm), prose +5%.
  Found on the way: the SwiGLU clamp symbols were set on the first GPU only, so the second stage's experts ran
  unclamped; fixed per stage (teacher-forced NLL 2.2459 -> 2.2305).
* 2026-10-26, v0.6.0 (`docs/DEEPSEEK-NEXT.md`): the expert cache adapts while serving by exchange (the leaving expert
  gets a host copy in a spare blob first), the routing-counts file is capped, and a teacher-forced verify-window bench
  measured what a drafter's rows cost: 0.43 of a plain round static, 0.13-0.19 with the exchange. Essay decode 19.3 ->
  29.7 tok/s, code 24.5 -> 32-36; quality unchanged.
* 2026-10-24, v0.5.2 (`docs/DEEPSEEK-NEXT.md`): the prompt attention runs on the tensor cores (a fused single pass,
  40 -> 9.5 ms per 2,048-token layer-chunk), the CPU tasks' rows go out compact (a 168 MB device-to-host copy per layer
  had stalled the compute stream 52 ms), decode's launches are fused (48 -> 28 a layer), and a PCIe lane pulls the next
  layer's predicted non-resident experts into spare cache slots while the layer runs (measured 17.6 tok/s against 19-20 with it off, so it ships off by default (`BNK_DS_PULL=1` turns it on)). Prompts 8,192
  tokens 467 -> 652 tok/s, 16,000 tokens 828 tok/s (groups of 8 chunks measured worse); decode 22.4 tok/s with the
  container on all 48 CPUs and the pool pinned one worker a core (default now); the copy test unchanged
  (ppl 1.037, top-1 99.61%). Project Maya's table IQ2_XXS CPU dot was bit-exact but slower on this Zen 2 (off).
* 2026-10-28: vision (`docs/VISION.md`): ISTA's mmproj and the image tokens' routing bias from their later commit
  (`a419d74c`), the encoder in src/vision, image cells in the prompt path (encoder rows, `exp_probs_b_vl`, no Engram);
  and the chat template of their repository (`templates/deepseek-v4.1.jinja`): the GGUF's left out the BOS token and
  put parallel tool results in arrival order (docs/DEEPSEEK-UPDATE-PLAN.md).
