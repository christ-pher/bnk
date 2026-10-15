# Next: the decode forward with far fewer launches (handoff, 2026-10-15 evening)

Read this first when resuming. It states where the engine is, what was measured today, the plan agreed with the
user for the decode path, and how to validate each step. Companion documents: `docs/PLAN.md` (the measured item
list), `docs/ROADMAP.md` (history), `docs/DUAL-GPU.md` (the pipeline), memory notes in Claude's project memory.

## Where things stand

* master `d3526b0` (pushed): the tensor-core prompt path (`src/kernels/moe_tc.cu`), 4,096-token chunks, decode
  fusions, stacked GEMVs. Server sweep (bnk_bench.py, batch 1): cold reads 4K 1,948 / 32K 2,741 / 64K 2,874 /
  128K 2,875 / 256K 2,653 tok/s; follow-up turns 571-946 ms; decode 105-122 tok/s. The production server was NOT
  running when the session started and is left stopped (`./run.sh iq3_s` starts it; `tmux` session "bnk" is the
  user's usual place for it).
* branch `tensor-parallel` (pushed, worktree `/opt/engines/bnk-tp`, commit `157b55a`): every layer split across
  both GPUs, output identical to the pipeline, but a verify round takes 28.7 ms against 22.5 and prompt reads
  1,100 tok/s against 2,000. Not merged. It becomes useful only after the change below.
* Zig rewrite: discussed and declined (the bottlenecks are on the GPU; the kernels would stay CUDA C++).

## The problem, in numbers (node-level nsys, one conversation, `bnk run --gpus all`)

A decode forward of ~3 rows on GPU 0 (layers 0-25) is 11.0 ms of GPU time and **972 kernel launches** (~37 per
layer); GPU 1 (22 layers) 10.1 ms. The two cards take turns (the pipeline), so a round is ~21 ms of GPU time plus
~2.3 ms of drafting: 2.8 tokens per round at 75-80% acceptance = 105-120 tok/s.

Per forward on GPU 0: dense GEMVs 5.8 ms (`gemv_mma_k` 150 x 16.7 us, `gemv_r_k` 144 x 14.3 us, `gemv_f_k` 129 x
9.6 us), routed experts 3.2 ms (`moe_gu_r_k` 52 x 82 us, `moe_down_r_k` 52 x 42 us), attention 0.85 ms, DeltaNet
recurrence 0.43 ms, routing 0.43 ms, 402 tiny kernels 1.56 ms. Kernel time sums to 12.6 ms against 11.0 busy:
the side streams overlap a little. Bytes per forward per card: ~2.0-2.2 GiB of dense weights + ~1.4 GiB of
experts, i.e. ~4 ms at HBM bandwidth. **We are at ~35% of the bandwidth bound**, and the gap is launch and ramp
latency of small dependent kernels, not any single slow kernel. Evidence that it is the chain, not the kernels:
halving every matrix under tensor parallel did not shorten the GEMVs (15 us stays 15 us); stacking pairs of
GEMVs gained 2%; HC up as Q8_0 gained 0%.

## The plan: a persistent per-layer kernel (then tensor parallel on top of it)

Goal: one launch per layer (48 per forward, not ~1,000), each a cooperative kernel that keeps every SM busy and
hands work between phases with grid-wide barriers, so the dependent chain costs barrier latency (~5-10 us per
phase on a V100) instead of kernel launch + drain + ramp per op. Target: a forward of ~6-7 ms per card (from
11), i.e. decode ~1.6x; then tensor parallel can split the work meaningfully.

Phases of one GDN layer (the common case, 36 of 48 layers), all rows of the window (T <= 8) at once:

1. HC pre-attention: norm of the 4-stream residual (hc_norm_q), down (10240 -> 320) + inject, silu, up (320 ->
   10240), mix -> `mixed` [T][2560] (+ its int8 / fp16 copies). Today 6 launches.
2. Projections from `mixed`: wqkv (10240 rows), wgate (6144), beta|alpha (96). Today 2-3 launches.
3. Conv + prep + recurrence (per head, sequential over T) + gated norm -> `gdn_n`. Today 4 launches.
4. ssm_out (6144 -> 2560) -> `out`; combine into the residual (fused into the next norm today).
5. HC pre-MoE (as 1) -> `mixed`.
6. Router + top-k + plan; shared expert gate|up, silu*up, down; routed experts gate/up, act, down (resident
   hits), reduce with the CPU misses. Today ~10 launches.

A persistent kernel per layer runs these as stages separated by `cooperative_groups::grid_group::sync()`
(launch with `cudaLaunchCooperativeKernel`; 80 SMs x 2 blocks of 256 threads = 160 blocks, every stage
distributes its row groups over the 160 blocks). The existing per-op device code is reusable: the GEMV row-group
body (`gemv_r_k`'s loop, `mma_weights` / `mma884` for several rows), the expert row loops, the norms. Each stage
becomes a `__device__` function `stage_X(block_id, n_blocks, ...)` taking the same pointers the kernels take now.

### Steps, in order (each measurable on its own)

1. **Feasibility micro-test** (half a day): a cooperative kernel doing stages 1+2 of one GDN layer (norm -> down ->
   up -> mix -> wqkv) with 4 grid syncs, on real weights of one layer, against the current 8-launch sequence.
   Measure us per layer at T = 1, 3, 5, 8. If it is not at least 1.5x faster for that chain, stop and rethink.
   Tools: copy the pattern of `tests/bench_gemv.cu` / `tests/bench_moe.cu`; time with events.
   Watch: cooperative launch inside CUDA graph capture is supported (cudaLaunchCooperativeKernel can be captured)
   but check it on CUDA 12.8 / driver 580 first; if not, the forward stays eager-launched (48 launches is fine).
2. **The HC block as one kernel** (`hc_pre`: 6 launches -> 1, twice per layer). Exactness: the quantized
   outputs must be bit-identical (`BNK_EXACT=1` tokens equal). Expected ~0.5 ms per forward per card.
3. **The GDN block as one kernel** (stages 2-4): projections + conv + recurrence + norm + ssm_out. The recurrence
   is per head (48 heads x 128x128 state) - 48 blocks busy, the rest idle during that stage; acceptable.
4. **The MoE block as one kernel** (stage 6): router/top-k/plan on one block, then gate/up for every (hit expert,
   row group) over the grid, barrier, act, barrier, down, barrier, reduce. The CPU-miss mailbox stays: the
   reduce stage spins on `seq_done` as `moe_reduce_k` does now (a grid sync does not wait on host memory).
5. **Attention layers** (12 of 48): QSA select + attention + wo; lower priority, they are 1/4 of the layers.
6. **Then tensor parallel again**: with 1 kernel per layer per card, split as on the branch (the split code is
   done); the exchange becomes a grid-sync + copy; expect the 20 -> ~13 ms the earlier estimate promised.

### Validation at every step

* Tokens: `build/bnk run --model $M --tokens-file $S/tok/follow4k.txt --ctx 262144 --slots 1 --gpus all --mtp
  /root/.cache/bnk/mtp/mtp-q2_0.gguf --draft 4 --min-p 0.8 --draft-vocab /root/.cache/bnk/draft-vocab-*.bin
  --max-new 64` must print the 67 tokens in the scratchpad's `ref_out.txt` (also in docs/ROADMAP.md's history:
  "1596 1144 4087 1156 579 3296 25 328 8160 369 ..."). Token files: regenerate with the scratchpad's
  `mkprompts.py` (uses /opt/benchmarks/bnk_bench.py's prompt builder) if /tmp was cleared.
* Speed: `bnk pbench` through the scratchpad's `pb.sh 4k,32k` (server settings), then the real sweep:
  `./run.sh iq3_s` in the background and `/opt/benchmarks/bnk_bench.py --no-publish --runs 1`.
* Exactness across batching: `tools/serve_check.py`, `tools/park_check.py` (both set BNK_EXACT=1).
* Quality: `bnk nll` with BNK_NLL_SKIP / BNK_NLL_REF (note: at 4K depth the metric is chaos-dominated: the three
  pre-existing expert paths disagree by up to 0.26 nats among themselves; compare the same path before/after).
* Profiles: `nsys profile -c cudaProfilerApi --capture-range-end=stop -t cuda --cuda-graph-trace=node` with
  BNK_PROFILE_DECODE=1 on `bnk run`; summarize with the scratchpad's `nsum.py` (per device: busy %, kernels by
  name, idle gaps). ncu works (`--set` sections) and showed the expert kernel latency-bound at 8 warps/SM.

### Risks and known traps

* `kMaxWindow = 8` is baked into ~78 sites; the persistent kernel should keep T <= 8.
* Cooperative launch needs all blocks co-resident: shared memory and registers per block bound the grid
  (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`); the IQ tables in shared memory (8 KB for IQ2_S) count.
* Never enable CUDA peer access on this host: `cudaMemcpyPeerAsync` then takes 49 ms per copy. Pinned host +
  events costs 15-28 us per exchange.
* The expert cache's `slot_of` uses -2 for "the other card's expert" on the TP branch; master has only -1.
* GPU 0 has ECC enabled with `pending: Enabled` (the disable did not register); V100 HBM2 ECC costs no
  bandwidth, so it is not a performance factor; `nvidia-smi -i 0 -e 0` then reboot if the user wants it off.
* Builds: `cmake --build build -j 32` in the repo; ptxas register report for a file:
  `nvcc -arch=sm_70 -O3 --use_fast_math --expt-relaxed-constexpr -std=c++20 -Xptxas=-v -c FILE -I src
  -I third_party/ggml/include -I third_party/ggml/src -o /dev/null`.
