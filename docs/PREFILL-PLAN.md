# Prompt processing on the DeepSeek engine: where the time goes and what to do (branch `prefill-gemm`, 2026-10-26)

Decode is settled for now (v0.6.0's exchange plus the DSpark drafter: 37 tok/s average, 70 peak in the agent's
session). This is the plan for the prompt path. `docs/DEEPSEEK-NEXT.md` has the profiles it builds on; `docs/DEEPSEEK.md`
describes the path (`src/deepseek/ds_prefill.cpp`: layer-major groups of up to 4 chunks of 2,048 tokens, non-resident
experts streamed through two 1 GiB staging buffers in waves, prefetched from the next layer's router, the CPU pool
taking what a cost model says, the two GPUs pipelined over groups).

## Two regimes, and which one the agent pays for

| Prompt | Measured | Bound by |
|---|---|---|
| 250-800 new tokens after a cached prefix (every agent turn: the tool result) | 60-110 tok/s (3-7 s a turn; `prompt_tokens` 152K with `reused` 152K in the 2026-10-25 session) | PCIe: ~2 GB of non-resident experts a layer a stage, whatever the chunk size, at ~12 GB/s; the GPUs stream in sequence (one group, no pipelining) |
| 2,048 tokens | 279-297 tok/s | PCIe, as above |
| 8,192 tokens | 652 tok/s (6.3 + 6.2 s, the stages in sequence) | the kernels: per stage 5.3 of 6.3 s, of which the experts' tensor-core GEMM 3.1 s at 22 TFLOPS, attention 0.8, dense GEMMs 0.8, the Engram gather 0.55 exposed on stage 0 |
| 16,000 tokens | 828 tok/s (two groups pipelined over the cards) | the kernels |
| 70,000 tokens (2026-10-26, with the drafter and the exchange) | 662 tok/s | the kernels |

The agent's prefix is cached between turns, so its prompt time is the first regime: a few hundred tokens a turn
routed to nearly every expert, which makes every layer stream its full non-resident set. A faster GEMM does nothing
there. Long uncached prompts (a new session's system prompt and files, a big paste) are the second regime, where the
GEMM is half the time.

So two tracks, measured first.

## Step 0: measure (one GPU window, ~30 minutes)

1. `BNK_DS_PF_PROF=1 build/bnk pbench ...` (or `run`) on prompts of 256, 1,024, 2,048 and 8,192 tokens with the current
   master (the exchange and the drafter changed the resident set and the budgets): the per-phase split (route, attn,
   gpu_pairs, cpu_pairs, staged, copy_mb, gather, engram_host) per stage. The 2026-10-24 numbers above are the
   reference.
2. The agent's real suffix sizes and prefill times from `/api/stats` `recent` (`prompt_tokens - reused`, `prefill_ms`)
   over a session, to weight the two tracks by what is actually paid.
3. The GEMM ceiling on this card at its 150 W cap: `bench_gemm` with cuBLAS fp16 on the expert shapes (M = tokens an
   expert sees in a group: 32-128 at 2K-8K; N = 2,304 and 5,120; K = 5,120 and 2,304), batched over 384 experts. If
   cuBLAS lands at 60-80 TFLOPS where `moe_tc_k` makes 22, track A is a 2-3x on that term; if it lands at 30, the
   kernel is near what Volta gives these shapes and track A shrinks to tuning.

## Track A: the experts' GEMM on long prompts (the 3.1 s of 6.3 s a stage at 8K)

`moe_tc_k` (`src/kernels/moe_tc.cu`) unpacks each expert's quantized rows into fp16 A fragments once per 64-token
tile and multiplies with `mma.m8n8k4` (Volta's instruction), one block per SM at 255 registers, the activation tile
staged in shared memory per K step. It measured 22 TFLOPS; the unpack is a small share of the instructions, so the
loss is occupancy and latency (8 warps a SM, the per-step staging), not the dequantization.

Two designs, decided by step 0.3:

* **A1, dequantize once, multiply with a real GEMM.** Per batch of experts, unpack their three matrices into an fp16
  scratch (an expert is 3 x 2,304 x 5,120 halves = 70 MB; 14 experts a GiB), then one batched tensor-core GEMM per
  matrix over the batch (cuBLAS `GemmBatchedEx` with the tokens of each expert gathered as its M, or a CUTLASS-style
  grouped GEMM with per-expert M), the SwiGLU and the down projection the same way. The unpack writes 70 MB an expert
  and the GEMM reads it back: at 900 GB/s that is 0.16 ms an expert, ~60 ms a layer-group of 384 experts, against
  the 158 ms the GEMM takes now, so the dequantization round trip is affordable if the GEMM itself reaches 60+
  TFLOPS. **The cost is VRAM**: the scratch (1 GiB) comes out of the expert cache, ~100 resident experts a GPU, about
  1-2 points of decode misses. The two 1 GiB staging buffers of the streaming path are an alternative home when the
  non-resident wave is done with them (they are double-buffered: one is free while the other streams).
* **A2, a better fused kernel.** Keep the unpack-in-kernel design and raise its occupancy: 128-token tiles (half the
  unpacks per token; the activation tile at 128 x 128 halves is 32 KB, two blocks a SM still fit in 96 KB), the A
  fragments of the next K step unpacked while the current step's MMAs run (software pipelining; the register budget is
  the limit at 255), and the row group doubled so each unpack feeds twice the MMAs. Expected 1.5-2x on the kernel if
  the latency hypothesis is right; nothing if the instruction issue is the limit. No VRAM cost. Measured as
  `test_moe_tc` timings on a real layer (the test exists: `tests/test_moe_tc.cu`, host reference, uneven token counts).

A2 first for a day, because it costs no VRAM; A1 if A2 stalls under 40 TFLOPS and step 0.3 says cuBLAS gets 60+.
Either way the Engram gather (0.55 s exposed on stage 0 at 8K) runs ahead of its group on the pool's threads: an
easy 8%.

## Track B: the per-turn suffix (PCIe-bound, the agent's actual cost)

The bytes a layer streams are set by how many experts the chunk routes to, and a 250-token chunk touches most of them.
The levers are residency and overlap, not kernels; none is a 2x.

1. **Residency.** 29% of the experts are resident. The exchange now keeps the hot ones for the served workload, which
   helps these suffixes too (they are the same distribution as decode). More VRAM for experts: the dense weights are
   ~6 GB a GPU at Q5_K / Q8_0; a Q4_K build of the dense tensors would free ~1.5 GB a GPU, ~150 experts each (+3.5
   points of residency), at a quality cost to be measured with the copy test (ppl 1.037 / top-1 99.61% is the bar).
   The compressed-KV context is mapped as used and costs little; the drafter's 1.19 GiB on GPU 1 could move its dense
   weights to Q8_0 (0.6 GB back) at no quality cost.
2. **Both links at once.** A single group runs stage 1 then stage 2, so only one GPU's PCIe link streams at a time.
   Stage 2 can stream its own layers' hottest non-resident experts *by rank* while stage 1 computes, as the prompt
   path already does for a group's first layer (`pf_prefetch(l0_, pf_rank_...)`): extend it to as many of stage 2's
   layers as its staging buffers hold, started when stage 1 starts. Bounded by the 2 GiB of staging, so it covers
   roughly the first layer's worth; a third buffer, or streaming straight into spare cache slots (the PCIe lane's
   mechanism, which exists), raises the bound. Expected 10-20% on short chunks.
3. **The CPU share.** The cost model's constants (`cpu_us_per_expert_token`, `pf_pcie_us_mb_`) predate the 48-CPU pinned
   pool; re-measure them, since for a 250-token chunk most experts see 1-4 tokens and the pool beats the link at
   those counts.
4. **Tiny suffixes through decode windows.** Below ~64 new tokens the decode path with the resident set and the CPU
   pool is as fast as streaming; `prefill_min` (24) is the switch. Measure where the crossover sits now.

## Order

1. Step 0 (measure), half a day with the server down for ~30 minutes.
2. Track B.3 and B.4 (constants and the crossover): an hour each, pure wins.
3. Track A2 (the fused kernel's occupancy), one to two days; the Engram gather overlap alongside.
4. Track B.2 (stage 2 streaming during stage 1), one day.
5. Track A1 only if A2 cannot reach 40 TFLOPS and the ceiling bench says the card can.
6. Track B.1's dense-weight requantization last: it trades quality and needs the copy test to clear.

## Validation

* `tests/test_moe_tc.cu` against its host reference for every kernel change; `build/bench_ds_attn`, `bench_gemm` for
  timings; `BNK_DS_PF_PROF=1` for the per-phase split.
* The copy test (`docs/DEEPSEEK-NEXT.md` Commands: `BNK_NLL_T=8 BNK_NLL_SKIP=4096 bnk nll ...`): ppl 1.037, top-1
  99.61% must hold; `BNK_NLL_REF` KL against a dump from before the change under 0.01 mean.
* End to end: `bnk run` prompts of 256 / 2,048 / 8,192 / 70,000 tokens, then the server with the agent for a session.

## Results (2026-10-26)

**Step 0, measured** (`pbench`: a 16K-token prefix cold, then a follow-up turn with the prefix cached; IQ2_XS, both GPUs,
the drafter loaded, the served counts): 64 new tokens 2.31 s, 250 4.01 s, 800 5.2-7.6 s, 2,048 6.50 s; cold 2K 6.19 s,
8K 11.8 s, 16K 18.1-18.6 s. The ceiling bench (`build/bench_expert_gemm`, cuBLAS fp16 batched over 64 experts at the
150 W cap): 34 TFLOPS at 128-512 tokens an expert (44 at most), 18 at 64, 10 at 32. So `moe_tc_k`'s 22 TFLOPS is half
of what the card gives these shapes: **track A is worth at most ~1.5x on that term (~15% on long prompts), A1's
dequantize-then-cuBLAS would lose to its own round trip.** Shelved behind track B.

**The finding: the prefetches, not the CPU constant.** A sweep on the follow-ups: `BNK_CPU_US_PER_EXPERT_TOKEN` 15
changed nothing (the model already calibrates `pf_cpu_scale_`); the guessed prefetch off took 64 tokens 2.31 -> 0.93 s,
both prefetches off 0.68 s. A prefetched expert counted as free on the GPU, so a short chunk streamed 9.4 MB experts
the CPU would have answered faster, on the critical path. Now (`pf_moe`, `prefill_group_here`): the guessed prefetch
takes only experts with at least `ceil(t_pcie / t_tok) - 1` guessed tokens (the ones the cost model streams anyway;
`BNK_DS_PF_GUESS_ALL=1` restores all), and the rank-ordered first-layer prefetch runs only for groups of
`BNK_DS_PF_RANK_MIN` (1,024) tokens or more.

| Prompt | Before | After |
|---|---|---|
| 64 new tokens (cached 16K prefix) | 2.31 s | **0.64 s** |
| 250 new tokens | 4.01 s | **1.52 s** |
| 800 new tokens | 5.2-7.6 s | **2.94 s** |
| 2,048 new tokens | 6.50 s | **4.98 s** |
| Cold 2,048 | 6.19 s | **4.28 s** |
| Cold 8,192 | 11.8 s | 12.0 s (noise) |
| Cold 16,000 | 18.1-18.6 s | 18.5-19.0 s (noise) |

Quality: teacher-forced NLL after a 3,000-token prefill, KL to master 0.0037 mean against 0.0036 for master against
itself. Thresholds swept: `BNK_DS_PF_RANK_MIN` 4,096 = 1,024 within noise, never (100,000) costs the 2K follow-up 1.6
s; the guessed prefetch off is 2-3% slower at 2K-8K than the selective one. Decode windows for a 64-token suffix
(`BNK_PREFILL_MIN=128`): 1.52 s, slower; the crossover stays at 24.

**Next on this branch**: B.2 (stage 2 streams its layers' hot experts while stage 1 computes; the 2K cases are still
PCIe-bound at 4-5 s), the Engram gather overlap on long prompts (~8% at 8K), then A2's occupancy work (<= 15%).
