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
  3.3x a token's bytes and gain nothing"), which is our case.
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
