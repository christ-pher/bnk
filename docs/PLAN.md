# Performance plan (from 2026-10-15)

Where the engine stands after the tensor-core prompt path (commits 97b61ed, a15ddfe; `docs/ROADMAP.md` has the
measurements), and the routes left for more speed without touching quality (fp16 K/V, no new quantization,
greedy output unchanged). Items are in the order they will be worked on.

| # | What | Expected | Effort | Status |
|---|---|---|---|---|
| 1 | **256K read inside a sweep**: 1,687 tok/s after the shorter depths vs 2,655 alone on a fresh server (and 2,650 in `bnk pbench`). Find the state the earlier conversations leave (suspect: the expert cache not grown back after a big conversation is parked, or spare pinned buffers) | up to 1.5x at 256K, likely a small fix | small | |
| 2 | **Expert kernel register diet** (`moe_tc.cu`): ncu shows 8 warps per SM, 40% issue, 4.9 cycles per issued instruction, latency-bound. Two blocks per SM without spills (the 128-register token-split try spilled and ran IQ2_S/IQ3_XXS layers 3x slower) | ~2x on the expert share, ~20-25% on cold reads | medium | |
| 3 | **Dense prompt GEMMs**: cuBLAS at ~46 TFLOPS with fp32 outputs (`Kernel2`, 64x64 tiles); fp16 outputs plus the convert folded into the next kernel, or bigger tiles | ~10% on reads | small | |
| 4 | **Decode**: (a) window 8 -> 16 rows for batched streams (roadmap item 6: more drafts verified per conversation when 2-3 decode together); (b) the IQ expert unpack (experts at ~half of HBM bandwidth, instruction-bound); (c) remaining small-kernel fusion (402 tiny launches per forward, ~12% of GPU time, mostly overlapped already) | (a) 10-20% with several streams; (b) 5-10%; (c) 3-5% | medium | |
| 5 | **Tensor parallel for one conversation** (both GPUs on every layer instead of taking turns; each card idles ~50% of a single-stream round today) | ~1.6x single-stream decode on paper | large: see below | exploring |

## Decode today, per forward on each GPU (~11 ms, node-level nsys)

Dense GEMVs ~45% (big ones at memory bandwidth), routed experts ~25% (~430 GB/s, IQ unpack bound), 402 tiny
kernels ~12% (fusing 5 per layer gained ~1%: they overlap on side streams), attention / DeltaNet ~10%; the
drafter ~10% of a round. One conversation: 2.9 tokens per round at 75% acceptance, 105-120 tok/s.

## Tensor parallel: what it would take

The layer pipeline runs GPU 0's 26 layers, then GPU 1's 22, for every verify window; each card is idle while
the other works. Tensor parallelism splits every layer across both cards so both work on every token:

* **Dense projections**: split by output rows (wq / wk / wv / wqkv / wgate / hc up ...) or by input columns (wo,
  ssm_out, hc down): row-split needs the full input on both cards (a broadcast), column-split produces partial
  sums (an all-reduce). Per layer: ~2 exchanges of T x 2560 floats (10 KB per token).
* **Routed experts**: each card keeps half of every layer's experts (the cache already holds every expert of
  its own layers; the split changes which half). A token's 10 experts land on both cards; the partial MoE outputs
  are all-reduced with the dense ones.
* **Attention / DeltaNet / PLE**: split by heads (24 attention heads, 48 DeltaNet heads, 2 KV heads: KV would be
  replicated or split 1/1), K/V cache per card halves.
* **The exchange**: no NVLink or peer access on this host, so every exchange is device -> pinned host -> device
  (two PCIe 3.0 x16 copies, ~5-10 us each for a few KB plus event latency), ~100 per forward: 2-3 ms of latency
  per forward against ~11 ms saved. The host must not sit in the loop: cudaMemcpyAsync between devices through
  pinned memory with events, captured in each card's CUDA graph.
* **Expected**: one conversation ~22 ms -> ~13-14 ms per round (1.6x) if the exchanges overlap nothing; the
  drafter (last stage today) stays on one card.
* **Cost**: every kernel launch site gets a row/column range, both engines' graphs run in lockstep with cross-
  device events, the expert cache's placement policy changes, parking / snapshots / batched lanes all follow.
  Prototype first: one layer's dense + MoE split across the two cards with the exchange, timed, before deciding.

Alternatives that keep the pipeline and still fill the idle card: none for a single stream (the next window
depends on the previous one's output). Two conversations already fill both cards (two lanes).
