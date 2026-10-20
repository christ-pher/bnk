# Ideas from 1Cat-vLLM (2026-10-19)

[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (being renamed 1Cat-LLM) is a vLLM fork whose purpose is making
modern models fast on V100 / SM70, including both families bnk runs (Qwen3.8-27B and Qwen3.8-Flash-Next). Apache-2.0,
so its code can be ported with attribution. This note records what it does, how it compares, and what bnk can take.
Reviewed at commit `6f3557c` (2026-10-11). Companion to `docs/PERF-IDEAS-2026-10-19.md`.

## Results (2026-10-20, branch `1cat-ideas`; details in docs/DENSE.md)

| # | Item | Outcome |
|---|---|---|
| 1 | DFlash2 drafter | **Done.** Q8_0 27B greedy code 108 -> 150 tok/s, maths 112 -> 156, prose and copies unchanged; served (temperature 1.0, thinking) 70.8 -> 82.1 tok/s (+16%) once drafts are sampled and verified by rejection sampling. Same tokens greedy. |
| 2 | 4-bit kernels | **Partly.** NVFP4 27B runs (19.8 GB, loader + GEMV global scales): served 95.7 tok/s vs 88.0 for Q8_0 with DFlash2; 1-row forwards -19%, 8-row only -5%: the tensor-core GEMV is latency-bound at 4-bit (28-41% of DRAM bandwidth). Cheaper unpack, prefetch, two accumulators: no gain; K split over 8 warps: -5% for every format. The remaining lever is a kernel that keeps more bytes in flight per warp at low register cost. NInfer (geoffwatts/ninfer-v100, a single-V100 engine) uses the same design (shift-built fp16, m8n8k4, split-K, no prefetch) and reports lower per-kernel bandwidth (344-372 GB/s at 4 rows on the same 50 MB gate/up) than bnk's (~455 GB/s at 8 rows); its 219 tok/s headline is a memorized corpus at 99% acceptance with context copying; on varied text it lists 43-93 tok/s on one V100. |
| 3 | FlashAttention-V100 | **Measured, not done.** Attention is ~37% of a 4K chunk at 28K depth, ~50% at 64K; the kernel is bound by shared-memory traffic (Nsight: L1/shared 70%, DRAM 0.7%, one 384-thread block per SM at 168 registers), not by K/V reads: 16 rows per K/V tile (RB=2) and 16-byte P/V loads changed nothing. A 2x kernel needs D split across warps (their design); worth ~15% at 32K, ~30% at 64K. Their kernel is per query head over [head][pos] K/V, without our gate or 8-bit KV: a port, not a drop-in. |
| 4 | Grouped QSA | **Already measured in bnk** (ROADMAP 2026-10-05): row sharing over the union of selections was slower than the pipelined one-row kernel bnk has (1.77x before); QSA attention is ~10% of a 78K read, so <= 8% is left. |
| 5 | 128-bit decode loads | **Already in bnk** (`kv_load8`, no paging). At 60K the decode attention moves ~470 GB/s per GPU; 64 splits is the best count (32/128/256 slower). <= ~2 ms per forward left at 60K. |
| 6 | Snapshots every N | **Done.** Every 8,192 tokens: a second ~20K-token request sharing the document 18.7 -> 6.8 s. |
| 7 | 16-row lookup windows | **Done.** Copy +27% (186 -> 237 tok/s), others unchanged (dense engine; the MoE engine keeps 8). |
| 8 | Process | Temperature 0.6 (per request; the configs keep the model's recommended 1.0): served DFlash2 82.1 -> 86.8 tok/s (3.10 per round), MTP 70.8 -> 76.1. A quality gate (Q3/Q4 in PERF-IDEAS) is still to build. |
| - | Sampled MTP drafts | Tried (the DFlash2 verification for the MTP): 70.4 vs 70.8 tok/s, not kept. |

## How it compares

Their headline numbers use **4 V100 SXM2 32 GB cards with NVLink** (TP4): twice bnk's memory bandwidth, and cheap
GPU-to-GPU exchanges.

| | 1Cat, 4 cards + NVLink | bnk, 2 PCIe cards |
|---|---|---|
| Flash-Next decode, MTP | 92-150 tok/s | 105-135 tok/s |
| Flash-Next 64K cold read | 6,241 tok/s (~1,560 per card) | 2,874 tok/s (~1,440 per card) |
| 27B decode, one stream | 182-306 tok/s (DFlash2) | 80-105 tok/s (MTP) |
| 27B 32-64K cold read | 3,567-4,069 tok/s (~1,000 per card) | 970 tok/s at 64K (~485 per card) |

On Flash-Next bnk already matches or beats them per card. Their lead is the dense 27B, from three things: the
drafter, the attention kernel and the 4-bit kernels.

## What to take, by value

1. **DFlash2: a better drafter for the 27B.** A separate 5-layer draft model (3.85 GB BF16,
   [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2); `DFlash2DraftModel`, hidden 5120,
   FFN 17408, 32 heads / 8 KV heads of 128, sliding-window attention in every layer). It reads the target's hidden
   states at layers 5, 19, 33, 47 and 61 (concatenated, projected by `fc` to 5120) and fills an 8-token block of mask
   tokens (`mask_token_id` 248070) in **one non-causal forward**: all 7 drafts in one pass instead of MTP's 7 steps.
   A selector (rank 256, top-16) narrows the vocabulary. Measured: 3.0 tokens per round on prose (bnk MTP 2.2-2.5),
   4.9 on code (bnk 3.6-3.9); round 16-17 ms on their TP4. Lossless: WikiText PPL 5.4993 vs 5.4994 without drafting,
   EvalPlus scores match. Expected for bnk: +25-35% on 27B decode before any kernel work. Fits the `Drafter`
   interface; needs hidden-state capture at five layers, a block (non-causal) attention mode, the selector. Weeks.
   Reference: `vllm/model_executor/models/qwen3_dflash.py`, `docs/design/sm70_dflash2_*.md`.
2. **TurboMind-style 4-bit kernels (our D2).** `docs/design/gguf_turbomind.md`: GGUF formats (Q4_0, Q4_1, Q8_0,
   Q4_K) become one canonical layout, unsigned 4- or 8-bit codes with an fp16 scale and additive min per 32 weights,
   run through TurboMind's tuned SM70 kernels (mma884, FP32 accumulation). Their first integration target is the
   Qwen3.8-27B Q4_K_M. Q4_K's nested scales are multiplied in FP32 and rounded to fp16, with reconstruction error
   reported against gguf-py. This is the path to the Q4 27B being ~2x faster than Q8_0, not 10% (measured
   2026-10-19: UD-Q4_K_XL 105 vs Q8_0 96 tok/s on today's kernels). Source: `csrc/sm70_turbomind/`. Also
   `docs/design/sm70_awq_exact_m5_batched_gemv.md`: their own exact-M5 GEMV beat TurboMind by only 10-13% and failed
   their quality gate, so TurboMind's kernels are the baseline to port, not to beat.
3. **FlashAttention-V100 for the 27B's long reads (our item 8).** D256 attention at 47-71 TFLOP/s; bnk's keeps the
   tensor cores 25% busy. Techniques: split D=256 into four D64 slices (paired warps share the QK work), K-stage
   ping-pong in shared memory, register prefetch and double buffering, split-KV in three, the visible prefix
   scheduled apart from the causal tail, GQA heads packed into wider QK/PV GEMMs. Measured 1.74x on a Q4096/KV64K
   operator. Expected for bnk: 1.5-2x on 64K+ cold reads. Standalone directory `flash-attention-v100/`.
4. **Grouped QSA for Flash-Next long reads.** Eight adjacent query rows select mostly the same KV blocks: load each
   block once for all eight, keep an exact per-row mask. 5.5x on the sparse attention, +30% full-model prefill at
   32-131K (their PR #387). Applies to FP4-Local and IQ3_S. Medium effort.
5. **128-bit loads in decode attention.** Reuse page metadata, one aligned 128-bit load instead of narrow ones: +21-26%
   on the operator, long-context decode +8% at batch 16, more on their E4M3 route (PR #268, #285). Check bnk's
   split-KV decode kernel; at 250K, attention is most of the 27B's round.
6. **Recurrent-state snapshots every N tokens.** They snapshot the hybrid (DeltaNet) state at fixed block boundaries
   ("Mamba align"), so any shared prefix is reusable: a 10K prefix 2.64 s cold, 0.16 s cached. bnk snapshots at two
   positions per request. One every 8K tokens extends prefix forking to shared documents (bounded by snapshot memory:
   66-82 MiB each).
7. **Wider verification with lookup ("q16").** Up to 16 lookup tokens verified per round: 316 tok/s on repeated
   context. That is bnk's item 11 (16-row window) plus the existing prompt lookup.
8. **Their process (our Q3 and Q4).** Every speed change ships with a quality gate: EvalPlus coding scores, GSM8K,
   a perplexity delta, natural-stop rates, sampled rather than greedy benchmarks, each number tied to a written
   "contract" (hardware, context, batch, KV type, sampling). One cheap detail: a temperature 0.6 coding profile
   raised draft acceptance 4.27 to 4.51 and slightly improved coding scores: a one-line config variant.

## Not to copy

- FP8 KV: bnk's 8-bit KV does the same job.
- NVLink-dependent paths (their custom all-reduce, multi-pair scheduling): bnk's cards have no working NVLink today.
- Broad model, image and video support: not a speed item.

## Order

1. Download the DFlash2 drafter and check its acceptance offline on our prompts before committing weeks to it.
2. In parallel, port the TurboMind 4-bit kernels for the Q4 27B.
3. Together these are the 27B's "hundreds of tok/s" path: lossless drafting on the Q4 file.
4. Then the attention kernel for long reads, then grouped QSA for the MoE models.
