# Performance plan (from 2026-10-15)

Where the engine stands after the tensor-core prompt path (commits 97b61ed, a15ddfe; `docs/ROADMAP.md` has the
measurements), and the routes left for more speed without touching quality (fp16 K/V, no new quantization,
greedy output unchanged). Items are in the order they will be worked on.

| # | What | Expected | Effort | Status |
|---|---|---|---|---|
| 1 | **256K read inside a sweep**: 1,687 tok/s after the shorter depths vs 2,655 alone on a fresh server (and 2,650 in `bnk pbench`). Find the state the earlier conversations leave (suspect: the expert cache not grown back after a big conversation is parked, or spare pinned buffers) | up to 1.5x at 256K, likely a small fix | small | not reproduced 2026-10-15: a second full sweep on a fresh server read 256K at 2,653 tok/s in sequence (1,687 once; 2,655 alone; 2,650 in pbench). Closed as a one-off; reopen if a sweep shows it again (`scratchpad/sample_stats.sh` samples the cache telemetry) |
| 2 | **Expert kernel register diet** (`moe_tc.cu`): ncu shows 8 warps per SM, 40% issue, 4.9 cycles per issued instruction, latency-bound. Two blocks per SM without spills (the 128-register token-split try spilled and ran IQ2_S/IQ3_XXS layers 3x slower) | ~2x on the expert share, ~20-25% on cold reads | medium | tried 2026-10-15: a WMMA version with the unpacked weights staged in shared memory (88-142 registers, two blocks per SM) was slower in both 16x32 and 32x32 warp-tile forms (32K cold 2,412 / 2,424 vs 2,740 tok/s): the fragment reloads from shared memory cost more than the latency they hide. The K-split kernel stays; a next attempt should keep its register-resident weights and get a second block per SM by halving the token tile instead |
| 3 | **Dense prompt GEMMs**: cuBLAS at ~46 TFLOPS with fp32 outputs (`Kernel2`, 64x64 tiles); fp16 outputs plus the convert folded into the next kernel, or bigger tiles | ~10% on reads | small | measured 2026-10-15 (`build/bench_gemm`): the big shapes already run at 70-81 TFLOPS with fp32 outputs; fp16 outputs help only the small ones (router, shared expert: 0.26 -> 0.17 ms per 4,096-token chunk), ~1% of a read; cublasLt heuristics no better. Closed |
| 4 | **Decode**: (a) window 8 -> 16 rows (scoped 2026-10-15: `kMaxWindow` is baked into ~78 sites: the MoE hit lists and mailbox sizes, the GEMV kernels' per-token register arrays (NT <= 8, the mma kernel's 8-token quad-pair groups), the commit graph tables [lane][first row][kept rows], the drafter's graphs and batched items, `PfItem`, the sampler's row buffers; a 2-3 day change, gains only with 2-3 streams) for batched streams (roadmap item 6: more drafts verified per conversation when 2-3 decode together); (b) the IQ expert unpack (experts at ~half of HBM bandwidth, instruction-bound); (c) remaining small-kernel fusion (402 tiny launches per forward, ~12% of GPU time, mostly overlapped already) | (a) 10-20% with several streams; (b) 5-10%; (c) 3-5% | medium | 2026-10-15: (c) partly done: matrices sharing an input are stacked at load and run as one GEMV with a split output (beta \| alpha, indexer k \| q, router \| shared gate, shared gate \| up, HC down \| inject, k \| v): exact, but only 2% (22.98 -> 22.53 ms per round): the side streams already overlapped those launches. The decode forward is bound by its chain of ~1,000 dependent kernels, not by any one of them: the next real step is a persistent per-layer kernel | |
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
* **The exchange, measured** (`build/bench_xfer`, 2026-10-15): device -> pinned host -> other device with a
  cross-device event costs 15.5 us for 10 KB (one token's 2,560 floats), 21 us for 40 KB (4 tokens), 28 us for
  80 KB; 96 exchanges per forward = 1.5-2.7 ms. `cudaDeviceCanAccessPeer` says the cards could talk directly, but
  with peer access enabled `cudaMemcpyPeerAsync` takes 49 ms per copy on this host (the PCIe root complex path):
  never enable it; the pinned-host route is the one to use. The host must not sit in the loop: the copies and
  events are captured into each card's CUDA graph.
* **Expected**: one conversation ~22 ms -> ~13-14 ms per round (1.6x): ~11 ms of the other card's turn saved,
  ~2 ms of exchanges added; the drafter (last stage today) stays on one card.
* **Cost**: every kernel launch site gets a row/column range, both engines' graphs run in lockstep with cross-
  device events, the expert cache's placement policy changes, parking / snapshots / batched lanes all follow.
  Prototype first: one layer's dense + MoE split across the two cards with the exchange, timed, before deciding.

Alternatives that keep the pipeline and still fill the idle card: none for a single stream (the next window
depends on the previous one's output). Two conversations already fill both cards (two lanes).
