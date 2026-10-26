# DeepSeek-V4.1-Flash: where things stand and what comes next (2026-10-24, after the second performance pass)

Read this first when resuming. `docs/DEEPSEEK.md` describes the model, the memory plan and the code; this file is
the state and the plan. The first pass (branch `ds-perf`, merged as v0.5.1) rewrote the prompt path and tuned the CPU
pool; the second (branch `ds-perf2`, merged as v0.5.2) put the prompt attention on the tensor cores, fused the decode launches and added
the PCIe lane for decode's missed experts. The numbers below are from it.

## Where things stand

* Serves correctly: `./run.sh ds-xs --port 8081` (IQ2_XS, 155 GB) or `./run.sh ds-xxs` (IQ2_XXS, 138 GB). Both
  backbones and the shared Engram file are in the Hugging Face cache (`tools/models.py resolve ds-xs`).
* Measured on the two V100s (IQ2_XS, 1 slot, ranked placement from the counts file, ~4,650 experts resident):

  | | before the pass | after (end to end, both stages) |
  |---|---|---|
  | Prompt, 2,048 tokens | 111 tok/s | 279 tok/s (3.8 + 3.6 s: PCIe-bound, ~2 GB of experts per layer per stage) |
  | Prompt, 4,096 tokens | 111 tok/s | ~400 tok/s (5.5 + 4.8 s) |
  | Prompt, 8,192 tokens | ~110 tok/s | 467 tok/s at v0.5.1; **652-655 tok/s** now (12.5 s: 6.3 + 6.2 s, ~1,300 tok/s per stage) with the tensor-core attention (467 -> 507) and the CPU rows' compact copy (507 -> 652); three runs agree |
  | Prompt, 16,000 tokens | | **828 tok/s** (19.3 s; two groups of 4 chunks pipelined over the cards). Groups of 8 chunks (`BNK_DS_PF_CHUNKS=8`): 606 tok/s, worse: one group leaves no overlap between the cards and costs resident experts; 4 stays the default |
  | Decode, plain | 12.95 tok/s (77 ms: 33 ms CPU experts, 38 ms GPU, 5 ms host) | 17.0-19.6 tok/s at v0.5.1; **19-20 tok/s** now on the 32-CPU cpuset (50-52 ms: 30-33 ms CPU experts, GPU kernels 21.4 ms with the launches fused); **22.4 tok/s** (44.7 ms: CPU experts 26.3, Engram rows 1.3) with the container on all 48 CPUs and the pool pinned one worker a core (`BNK_CPU_PIN`, now the DeepSeek default). The PCIe lane (`BNK_DS_PULL`) is off by default: 17.6 tok/s with it on (CPU 26 ms, but the GPU waits 28 ms instead of 15.6). The Engram row gather took 50 ms a token in three runs (2 ms in the others): scheduler placement of its per-token threads, now run on the pool's workers |
  | Decode, copying text (lookup drafts) | 12.25 tok/s | 32.3 tok/s (4.7 tokens per verify window, 126 of 133 drafts accepted) |
  | Copy test after a 4,096-token prefix | ppl 1.052, top-1 99.4% (master) | ppl 1.036, top-1 99.6%; KL vs master 0.02 (one outlier row) |

* The prompt path (`src/deepseek/ds_prefill.cpp`) is **layer-major**: a group of up to `BNK_DS_PF_CHUNKS` (4) chunks of
  `--prefill-chunk` (2,048) tokens goes through each layer before the next, so a non-resident expert crosses PCIe once per
  layer per group, not once per chunk (Strata's trick; "8K prompt 6x" in their notes). The non-resident experts stream
  through two 1 GiB staging buffers in waves; the next layer's experts are prefetched during this layer's attention from
  the next layer's router applied to this layer's FFN input (`BNK_DS_PF_GUESS=2` buffers; the first layer of a group by
  rank, `BNK_DS_PF_PREFETCH=1`); the CPU pool takes what a cost model with measured constants says it should (little: the
  CPU does ~300 us per (expert, token), the 2-bit decode is instruction-bound). Groups are pipelined over the two GPUs
  when a prompt has several. Buffers: the group costs ~1.5 GiB of VRAM per GPU over the old chunk path (fewer resident
  experts: ~3%).
* The prompt attention runs on the tensor cores (`attention_tc_k`, `BNK_DS_ATTN_TC=0` the fp32 kernel): a block takes
  32 heads of one token (the 64 query heads share the key rows), the 640 rows (window + selected) go by in tiles of 32
  staged once in shared memory and used for both products, the online softmax rescales the O accumulators through a
  fragment row map read at the start, q in fp16 after a power-of-two scaling (logit error ~1e-4). 40.2 -> 9.5 ms per
  2,048-token layer-chunk (`bench_ds_attn`); max error 1e-4 relative in `test_ds_ops`. Project Maya's three-phase
  Volta design (S into shared memory, softmax, P V; fp16 high + low parts) was the first attempt: 17-21 ms, bound by
  shared-memory traffic and 136 block-wide syncs per block, which is why the fused single pass replaced it.
* Decode's launches (60 -> 28 a layer; GPU kernel time 23.5 -> 21.4 ms a token, 1,912 -> 1,110 launches): the grouped
  output projection as one block-diagonal GEMV (`GemvSplit::grp_rows`: 8 x (quantize + GEMV) -> 2 launches), the
  hyper-connection pre-mix + rmsnorm + quantization in one kernel (`ds_hc_pre_norm_q`), the small-window Sinkhorn
  mixes' two kernels as one (the last of a token's 25 blocks runs the finish), rope + kv_prep as one, DeepSeek's
  routing + the plan as one (`moe_route_plan_ds`). What is left on the GPU side is near its floor: the dense GEMVs
  read ~145 MB of weights a layer (6.4 ms a token at HBM speed), attention 3.8 ms, the resident experts 3.2 ms.
* Decode: the CPU pool uses dynamic work ranges (`BNK_CPU_CHUNKS`, default 8 for this engine: 0.91 -> 0.64 ms per
  3-expert step), Engram rows are read on 8 threads (`BNK_DS_EG_THREADS_DECODE`; a cold table costs ~100 us per row in page
  faults), and prompt-lookup drafting runs through a drafter that never drafts itself (`LookupOnlyDrafter`,
  `BNK_DS_LOOKUP=0` off; copying text: 32 tok/s). DSpark is not ported: Strata skips it where most experts live outside VRAM ("verify rounds read
  3.3x a token's bytes and gain nothing"), which is our case; measured 2026-10-26 below (a verified row costs 0.43-0.48
  of a plain round here).
* Tests: `build/test_ds_ops`, `build/test_ds_gemv`, `build/test_ds_cpu MODEL [layer]` (the CPU pool vs a reference),
  `build/bench_ds_cpu MODEL [layer] [experts] [per call]` (pool speed, decode and verify shapes), `build/bench_ds_attn
  [T]` (the attention kernel on a chunk). `BNK_DS_PF_PROF=1` prints a per-phase breakdown of every prompt group.

* **The PCIe lane for decode's missed experts** (`BNK_DS_PULL=P`, default 0 = off): in layer il, layer il + 1's router
  is applied to il's FFN input (Strata's guess; 86% of the picks are routed), its non-resident picks are pulled by a
  16-block kernel from the store's host memory (registered mapped; a kernel reads it at the link's 13 GB/s, like the
  copy engine) into spare R-layout slots at the end of il + 1's cache region (`ExpertCache::reserve_spare`, repacked on
  the pull stream), and il + 1's device slot table is patched so the plan kernel finds them resident: the CPU is
  spared them (CPU experts 33 -> 26 ms a token at P = 1, 22 ms at P = 2). Non-blocking: the plan takes what has landed
  (a per-layer state 0/1/2), the restore after the FFN waits only for a straggler (none seen: the copy takes 0.75 ms of
  a ~1.5 ms window), the pull stream rejoins at the end of the forward. `bnk run` prints pulled / routed / late.
  **Off by default**: 17.6 tok/s against 19-20 without it (chain of builds: 13.3 blocking + 512-block copy, 16.8 small
  low-priority copy, 17.0 non-blocking, 16.4 spare slots, 17.6 pull launched after the plan). The trace (nsys, scratch
  `nsys_d10_pull1`): the copy itself is 0.75-0.84 ms and never late; the routing + plan kernel runs 223 us instead of
  26 whenever a pull is on the link (its mailbox writes to host memory and the system fence queue behind the pull's
  reads), the repack costs 170 us of SM time a layer on the side stream, and the hits kernels run ~2x slower beside it.
  The CPU saving is real (33 -> 26 ms a token at P = 1, 22 at P = 2), so the lane pays once the plan kernel stops
  talking to host memory while a pull runs: e.g. the plan writes the mailbox into device memory and a tiny kernel after
  the pull's window copies it out, or the mailbox goes through the pull stream's own gap.
* **The CPU side, measured** (EPYC 7402: Zen 2, 24 cores, AVX2 only; 8 x DDR4-2400): the pool reads 34 GB/s of weights
  against ~100+ GB/s of memory bandwidth, so it is instruction-bound in ggml's IQ2_XXS / IQ2_XS dots (the "IQ2_XS" GGUF's
  experts are ~85% IQ2_XXS by bytes). Project Maya's signed-grid table variant of the IQ2_XXS dot (`BNK_CPU_XXS_CT=1`,
  mdot.cpp) is bit-exact but slower here (0.44 vs 0.34 ms an expert: the 256 KB table shares the 512 KB L2 with the
  weight stream). The LXC's cpuset holds 32 of the host's 48 logical CPUs, covering 22 of the 24 cores: whole-core pairs
  for the container, and a pool that uses SMT without spinning, are the untested CPU levers; `BNK_CPU_PIN=1` now pins
  one worker per physical core of the allowed set first (unmeasured: see below).
* Measurement hygiene: compiling on the host during a run slows the prompt path (two of today's runs read 2K prompts
  at 185 tok/s instead of 290 while nvcc ran); the per-phase profiler `BNK_DS_PF_PROF=1` synchronizes after every phase
  and understates prompt speed by ~15%.

## What the profiles say (nsys, 2,048-token chunk)

* **The 8K prompt's bubble, found and closed.** The nsys timeline of an unprofiled run (scratch `pf_timeline.py`)
  showed each layer's MoE starting ~51 ms after the routing: a 168 MB device-to-host copy of every token's FFN input
  for the CPU tasks, on the compute stream, at 3.2 GB/s beside the expert stream, with everything behind it waiting
  (~1 s per stage). Now only the tokens with CPU tasks go out, gathered compact on the device first
  (`ds_rows_gather`, `h_cpu_tok` / `h_cpu_map`; a quarter of the group's rows of room, else the whole copy): 8K
  507 -> 652 tok/s, both stages 6.2-6.3 s with 0.3 s of expert wait. What the per-phase profiler had shown as a slow
  stage 1 was the profiler's own synchronization. Still exposed on stage 0: ~0.55 s of Engram gathering (the loose end
  below). Per stage the kernels now take ~5.1-5.6 s of the 6.3: attention 0.8, the experts' tensor-core GEMMs ~3.1,
  dense GEMMs 0.8.

* Per layer-chunk of 2,048 tokens: attention_k 30 ms (fp32, one key at a time, every head re-reads the key row),
  moe_tc (the experts on the tensor cores) ~39 ms at 22 TFLOPS, the dense GEMMs ~10 ms, Sinkhorn mixes 1.8 ms. Two
  mistakes found and fixed: `Gemm::run` converted strided inputs one row per launch (16,000 launches per layer for the
  grouped output projection; 65 ms of host-bound time), and the compute stream's small H2D copies (routing tables, the
  CPU answer) queued behind the 2 GB prefetch in the copy engine's FIFO, stalling the compute stream until the prefetch
  was through (the layer took 320 ms wall for 114 ms of kernels). Those copies are kernels over mapped memory now.
* Decode (77 ms per token before, 59 after): 40 layers x (GPU ~1 ms -> 0.4 ms + CPU experts ~0.8-0.9 ms), serialized
  by the dependency. The GPU part was attention_k 0.5 ms (8 blocks for one token), hc_mixes 2 x 85 us (one SM reading
  the 2 MB mixer), dense GEMVs 0.2 ms (near HBM bandwidth), ~50 small kernels; the first two are fixed. Profile with
  `nsys ... --cuda-graph-trace=node` (without it the graphs' kernels are invisible).

## Speculative decoding: what a verify window costs here (measured 2026-10-26, branch `oracle-drafts`)

Would a real drafter (DSpark) pay on this machine? The verify window's cost decides it, so it was measured directly
rather than inferred: `BNK_ORACLE=FILE BNK_ORACLE_WS=1,2,3,5,8 build/bnk run ...` decodes a prompt once plain
(W = 1, greedy; writes the prompt and its 400 output tokens to FILE), then once per window size with the reference's
next tokens as the drafts, **teacher-forced**: the window's tokens are kept whatever the argmax says
(`Generator::oracle_`, `BNK_ORACLE_OFF` guards the reference run), so every round is a full W-row window of the
model's own text and the per-row cost is read straight off the round time. IQ2_XS, both GPUs, the server's flags, the
served ranking file (a copy), one Python module and one essay (`tools/bnk_main.cpp` prints the usual decode report
per W plus how often the argmax agreed with the forced rows).

| rows per window | code: ms/round, tok/s, CPU experts ms, misses/forward | prose: ms/round, tok/s, CPU experts ms, misses/forward |
|---|---|---|
| 1 (plain) | 40.9, 24.5, 24.0, 124 | 51.9, 19.3, 35.2, 191 |
| 2 | 59.1, 33.8, 41.0, 248 | 76.8, 26.0, 59.0, 381 |
| 3 | 76.7, 38.9, 54.5, 370 | 100.4, 29.7, 78.6, 570 |
| 5 | 108.8, 45.4, 82.7, 614 | 147.1, 33.6, 121.3, 943 |
| 8 | 157.4, 49.8, 124.4, 972 | 211.2, 37.1, 178.8, 1,498 |

* **A verified row costs 17-18 ms on code and 23-25 ms on prose: 0.43-0.48 of a plain round, flat from W = 2 to 8.**
  The misses per forward scale exactly with W (adjacent tokens share no missed experts), and the CPU expert time
  accounts for all but ~1 ms of each row: the GPU side of a window amortizes, the host side does not. Strata's "verify
  rounds read 3.3x a token's bytes and gain nothing" is this. The live fit over the day's 145 agent requests (150K
  context, 20-27% misses) gave the same slope, 21.8 + 17.9 ms x tokens per round.
* **Break-even acceptance per draft token** (expected tokens 1 + a + a^2 ... against the plain round), code: W = 2
  needs a >= 0.45, W = 3 0.56, W = 5 0.68, W = 8 0.77 with a drafter that costs nothing; prose within 0.02 of these.
* **DSpark as shipped** (DeepSeek-V4.1-Flash-1017's own safetensors, shards 44-46 of 48; there is no GGUF of it, ISTA's
  repository has none): 3 MoE layers (`mtp.0-2`) of **128 routed experts, top-3**, moe inter 2304 (an expert is the
  backbone's size, 35.4M parameters), a shared expert, `main_proj` [5120, 15360] from three target layers' hidden
  states, block size 5 (one forward over anchor + noise tokens drafts 5 positions), a Markov head (rank 256) and a
  confidence head. 13.6B expert parameters: 3.9 GB at IQ2_XS, 7.6 GB as shipped (fp4); 0.57B dense. The DSpark
  paper (arXiv 2607.05147) reports accepted lengths of 4.9-5.4 on code with block 7 at temperature 1.0 on Qwen3-4B,
  and in DeepSeek's production (experts in HBM) budgets of 4-6 drafts and +51% throughput over MTP-1 for V4-Flash.
* **Modelled here** (`tools/ds_spec_scenarios.py`: drafter cost D per round, cache loss when its experts take VRAM): with the
  experts in RAM through the CPU pool (54 expert reads a draft forward, ~12 ms) a = 0.8 gives x1.15 on code, a = 0.9
  x1.36, a = 0.7 x1.02; with them resident (3.9 GB = ~1,100 backbone experts evicted: by the counts file the production
  miss rate goes 20.7% -> 27.9%, CPU work x1.35) x1.06 / x1.22 / x0.95. Prose: the same within 2%.
* **Verdict at the time: not now.** A port of the head (its GGUF conversion, 3 MoE layers with mHC and a 128-token
  window, the noise block, the Markov head, rejection sampling of 5 drafts) would buy 10-20% on code at an optimistic
  acceptance and nothing on prose, with the placement static. Everything that cuts the CPU expert cost per row raises
  the drafter's value proportionally; the adaptive exchange (next section) did exactly that the same day and the
  sweep re-run there puts the drafter at x1.35-1.9. The bench stays (`BNK_ORACLE*`, inert otherwise).
* **Side findings.** (1) The expert ranking is content-bound: fresh code misses 52% of its experts and an essay 80%
  against 11-27% inside the long agent session that wrote the counts; the plain round is 41 ms on code, 52 ms on prose.
  Routing counts from a mix of workloads would serve prose better. (2) Greedy does not reproduce across window
  shapes: the W-row verify path's argmax differed from the single-row reference's in 0.5-4% of rows (first
  disagreement per window: 99.5% / 98% / 97.5% / 92.6% of rows agreed at W = 2 / 3 / 5 / 8 on code, 96 / 97 / 94 /
  89% on prose), the kernels' accumulation order. Harmless at temperature 1.0; it is why the first, non-forced version
  of this bench diverged from its reference within a few tokens.

## The adaptive expert exchange (2026-10-26, branch `adaptive-experts`)

The verify-window sweep above exposed the real lever: the expert placement was static and ranked by the routing
counts of everything ever served (mostly the coding agent), so fresh code missed 52% of its experts and an essay 80%,
against 11-27% inside the agent's own session. The Qwen engine's adaptive cache could not be used as it was: its
swap unmaps the leaving expert at once, and here a resident expert has no host copy. `ExpertCache::adapt_exchange`
(expert_cache.cpp) does it as an exchange: the leaving expert first gets a host copy in one of the store's two spare
blobs a layer (`ExpertStore::take_blob` / `assign` / `release`; a helper thread writes it from the mapped model
file), then the slot is unmapped, refilled from the incoming expert's host blob on the copy stream and mapped, and the
incoming expert's blob becomes the spare. Between forwards only (`DsEngine::fwd_end`, every `BNK_DS_ADAPT` = 4
forwards, up to `BNK_DS_ADAPT_SWAPS` = 8 in flight a stage; off with the PCIe lane, whose host pointer table would go
stale), so the CPU pool and the prompt path never see a half-moved expert; the prompt path's load-time prefetch list
skips experts that became resident. The routing-counts file is capped (`merge_counts`, `BNK_COUNTS_CAP` = 2^28
routings, halved past it) so the next start ranks by the recent workload. `tests/test_expert_counts.cpp` covers the
merge and the ranking. Measured (IQ2_XS, both GPUs, the served ranking file, `BNK_MISS_TRACE=100`):

| | exchange off | exchange on |
|---|---|---|
| Essay, misses by 100-token stretch | 79, 79, 79, 79, 79, 77% | 66, 48, 42, 34, 31, 29% |
| Essay, tok/s by stretch | 19.1-19.4 throughout | 21.6, 25.2, 26.6, 28.3, 29.5, 29.7 |
| Python module, misses by stretch | 52% flat (the sweep's W = 1 runs) | 48, 36, 33, 28, 26, 24, 22, 20, 20, 20, 22% |
| Python module, tok/s by stretch | 24.5 | 24.7, 28.6, 29.3, 30.5, 30.8, 31.0, 32.9, 32.4, 35.9, 32.2, 32.4 |
| Exchanges a 100 tokens | 0 | 105-154 (both stages), i.e. ~1.2 GB/s of blobs each way at the start |
| Teacher-forced NLL of 4,000 tokens of docs (windows of 5) | 2.2459, ppl 9.449, top-1 50.66% | 2.2438, ppl 9.429, top-1 50.56%; misses 46.1% -> 27.4% over the pass, 1,176 exchanges |
| KL from the exchange-off pass (top-64 log-probs a row) | a second exchange-off pass: mean 0.0059, max 7.8, same top-1 97.12%; 2 rows over 1 (the run-to-run noise: CPU and GPU accumulate differently) | mean 0.0091, max 8.6, same top-1 96.57%; 3 rows over 1 |

* **Steady state is 20-30% misses on content the ranking never saw**, reached within 500 tokens, for +50% decode speed
  on prose (19.3 -> 29.7 tok/s) and +30-45% on code (24.5 -> 32-36). The counts file was the agent's; a mixed file
  will start closer. Long sessions pay nothing extra: an exchange is a 10 MB host memcpy on the helper thread and a
  10 MB PCIe copy plus a repack on the side stream.
* **Quality is unchanged**: the NLL matches within noise and the KL to the static placement is the kernels' CPU/GPU
  placement noise (the control row: 0.0059 between two identical static passes against 0.0091 with the experts moving between the devices). A wrong blob would have shown as a jump in NLL.
* **Spare blobs**: 2 a layer, 80 blobs, ~0.8 GB of pinned RAM. With one exchange a layer in flight at a time (the
  planner proposes at most one per layer per step) one spare would do; the second lets the next one start while the
  first's blob is still being read.

* **The verify-window sweep again, exchange on** (the same code prompt, the five W runs in one process, so the cache
  keeps adapting through them: misses 37% -> 23 -> 18 -> 16 -> 14%):

  | rows per window | ms/round | tok/s | CPU experts ms | misses/forward |
  |---|---|---|---|---|
  | 1 | 36.2 | 27.6 | 18.4 | 89 |
  | 2 | 40.8 | 48.9 | 20.9 | 110 |
  | 3 | 49.4 | 60.4 | 24.2 | 132 |
  | 5 | 87.5 (an outlier run) | 56.4 | 33.0 | 188 |
  | 8 | 84.2 | 93.1 | 44.9 | 269 |

  **A verified row now costs 5-7 ms, 0.13-0.19 of a plain round** (17-18 ms and 0.43 before): the misses per forward
  no longer scale with W because the window's tokens hit the same resident experts. That reopens the drafter question
  the section above closed. Re-modelled (`tools/ds_spec_scenarios.py 36.2 6.5 0.5`, DSpark's experts in RAM, 12 ms
  a draft forward): acceptance 0.7 gives x1.35, 0.8 x1.65, 0.85 x1.86 at windows of 4-6; break-even at W = 3 is
  a >= 0.45 with that drafter cost. The sweep's replay of one text four times over is a best case for the cache, but
  the traced fresh-code run sat at 20% misses from token 600 on, close to this. **DSpark is worth porting once the
  exchange is in production**, after a week of real sessions says where the steady-state miss rate lands.

## Next steps, in order

1. **Decode's CPU experts** (~20 of a ~38 ms token on 2026-10-25: 26.1-26.4 tok/s plain with the container on all 48
   CPUs and the counts ranking matured, misses 39%). The pool is instruction-bound in ggml's IQ2_XXS / IQ2_XS dots at
   a third of the memory bandwidth. Measured and settled: 24 threads pinned one a core is the optimum (23: 25.2 tok/s,
   22: 25.7, 20: 23.9, 16: 22.1, unpinned 24: 24.2; 48 spinning threads slower still), so Project Maya's "leave cores
   free" finding (their #433) does not apply here: our pool already leaves the SMT siblings to everything else. The
   lever left is a faster one-token IQ2_XXS / IQ2_S dot on Zen 2 (the signed-grid table variant lost: L2).
2. **The PCIe lane** (above) is not rescued by taking the mailbox off the SMs. Tried 2026-10-25 (`BNK_DS_MAIL_DMA`,
   not kept): the plan kernel wrote its request into a device mirror, the copy engines moved the request and the
   CPU's answer, the reduce read device memory. Lane off: 25.3 tok/s (3 copy nodes a layer cost 3-4%). Lane at P = 1:
   18.4 tok/s - the CPU experts fell 20.6 -> 16.3 ms a token as predicted, but each stage's GPU time rose 18.5 -> 26.3
   ms a forward (+0.4 ms a layer, misses 39 -> 28%). So the loss is a GPU-wide slowdown of the forward's kernels while
   a pull kernel streams host memory, not the mailbox's host traffic. The lane pays only if the pull runs while the
   GPU is otherwise idle (it is, during the CPU phase, but the pull outlasts it), or through a path that leaves the SMs'
   memory traffic alone. Shelved.
3. **The Engram gather's 50 ms mode**: three runs paid 50 ms a token in the row gather against 2 ms in the others, by
   run, not warming up. Not the NVMe (a cold random 4 KB read is 113 us and a token touches few rows; `MADV_WILLNEED`
   changed nothing): the gather spawned 8 threads per token, and a thread that lands beside a spinning pool worker waits
   out a scheduler slice. It now runs on the pool's own threads (`CpuExpertPool::parallel`; `BNK_DS_EG_THREADS_DECODE=N`
   restores the spawned threads). Verify over several runs that the 50 ms mode is gone. The prompt path's gather (0.55 s
   exposed on stage 0 at 8K) could run ahead of the group.
4. **The prompt path's floor**: both stages are ~5.3 of 6.3 s in kernels at 8K: the experts' tensor-core GEMMs ~3.1 s
   (22 TFLOPS), attention 0.8, dense GEMMs 0.8. 2K prompts stay PCIe-bound (~2 GB of experts a layer a stage).
5. **The IQ2_XXS build (ds-xxs) does not load** ("R layout: unsupported format"): 28 of its 40 down and 16 of its 40
   gate/up expert tensors are IQ1_M, which `rfmt.cpp` and the MoE kernels lack. Dropped by decision on 2026-10-25: no
   speed to gain (the pool is decode-bound and an IQ1_M expert decodes slower than an IQ2_XXS one even in Maya's
   optimized kernel) and lower quality; the 17 GB it saves is RAM, not time.
6. **Loose ends**: the indexer's `weights_proj` scale (1/64) is not applied (ranking-invariant); `--slots 2` untested;
   `forward_pipe` untested; vision not ported; the Qwen engine's prefill may queue its small copies behind the expert
   prefetch too (the copy-engine finding of v0.5.1).

## Commands

```bash
cd /opt/engines/bnk               # master holds v0.5.2 (both passes merged); branch work goes in a worktree
cmake --build build --target bnk test_ds_ops test_ds_gemv test_ds_cpu bench_ds_cpu bench_ds_attn -j 24
CUDA_VISIBLE_DEVICES=1 ./build/test_ds_ops && CUDA_VISIBLE_DEVICES=1 ./build/test_ds_gemv
./run.sh ds-xs --port 8081        # production (master, 8080) must not be running: both GPUs are needed
# copy test: a text and its first 512 tokens again, the prefix through the prompt path, the copy scored at 8-token windows
BNK_NLL_T=8 BNK_NLL_SKIP=4096 ./build/bnk nll --model "$M" --tokens-file tok_long4096_copy512.txt --ctx 16384 --threads 24 --gpus 0,1 --counts ~/.cache/bnk/counts-DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS.bnkc
BNK_DS_PF_PROF=1 ./build/bnk run --model "$M" --tokens-file tok_long8192.txt --max-new 32 --ctx 16384 --threads 24 --gpus 0,1 --counts ...
```

Stop only processes you started, by PID (`pgrep -af 'bnk run'`, `kill PID`); a `pkill -f` pattern matches the shell
running the command itself.
