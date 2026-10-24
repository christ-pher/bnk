# DeepSeek-V4.1-Flash: where things stand and what comes next (2026-10-24, after the first performance pass)

Read this first when resuming. `docs/DEEPSEEK.md` describes the model, the memory plan and the code; this file is
the state and the plan. The first pass (branch `ds-perf`) rewrote the prompt path and tuned the CPU pool; the numbers
below are from it.

## Where things stand

* Serves correctly: `./run.sh ds-xs --port 8081` (IQ2_XS, 155 GB) or `./run.sh ds-xxs` (IQ2_XXS, 138 GB). Both
  backbones and the shared Engram file are in the Hugging Face cache (`tools/models.py resolve ds-xs`).
* Measured on the two V100s (IQ2_XS, 1 slot, ranked placement from the counts file, ~4,650 experts resident):

  | | before the pass | after (end to end, both stages) |
  |---|---|---|
  | Prompt, 2,048 tokens | 111 tok/s | 279 tok/s (3.8 + 3.6 s: PCIe-bound, ~2 GB of experts per layer per stage) |
  | Prompt, 4,096 tokens | 111 tok/s | ~400 tok/s (5.5 + 4.8 s) |
  | Prompt, 8,192 tokens | ~110 tok/s | 467 tok/s (8.7 + 8.8 s; 944 / 926 tok/s per stage; the Engram rows gathered on threads under the first layers) |
  | Decode, plain | 12.95 tok/s (77 ms: 33 ms CPU experts, 38 ms GPU, 5 ms host) | 17.0-19.6 tok/s (51-59 ms: 30-37 ms CPU experts depending on the miss rate, 17 ms GPU, n-gram rows 2 ms); prose ppl 2.84 at 1-token windows (was 2.88) |
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
* Decode: the CPU pool uses dynamic work ranges (`BNK_CPU_CHUNKS`, default 8 for this engine: 0.91 -> 0.64 ms per
  3-expert step), Engram rows are read on 8 threads (`BNK_DS_EG_THREADS_DECODE`; a cold table costs ~100 us per row in page
  faults), and prompt-lookup drafting runs through a drafter that never drafts itself (`LookupOnlyDrafter`,
  `BNK_DS_LOOKUP=0` off; copying text: 32 tok/s). DSpark is not ported: Strata skips it where most experts live outside VRAM ("verify rounds read
  3.3x a token's bytes and gain nothing"), which is our case.
* Tests: `build/test_ds_ops`, `build/test_ds_gemv`, `build/test_ds_cpu MODEL [layer]` (the CPU pool vs a reference),
  `build/bench_ds_cpu MODEL [layer] [experts] [per call]` (pool speed, decode and verify shapes), `build/bench_ds_attn
  [T]` (the attention kernel on a chunk). `BNK_DS_PF_PROF=1` prints a per-phase breakdown of every prompt group.

## What the profiles say (nsys, 2,048-token chunk)

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

1. **Tensor-core attention for the prompt path** (a day). One shared KV head means a token's 64 query heads attend
   to the same 640 keys (128 window + 512 selected): S = Q[64 x 512] . K^T and O = P . K are real GEMMs per token
   (heads as the M dimension), flash-attention style over key tiles of 64 with the online softmax per head row.
   `mma.cuh` has the m8n8k4 helpers. Expected 30 ms -> ~5 ms per layer-chunk: ~25% of the prompt path's GPU time,
   and with long groups (where PCIe is hidden) the biggest remaining item. `bench_ds_attn` times it.
2. **Decode GPU side** (half a day): ~50 launches per layer. Fuse the 8 grouped `o_a` quantize+GEMV pairs into one
   launch each, the two `hc_mixes` + `hc_pre` + rmsnorm into one, route + plan is already fused in the Qwen engine
   (`moe_route_plan`): use it. Each 5 us launch saved is 0.2 ms per token.
3. **Decode CPU side**: the IQ2 decode runs at ~1.7 weights per cycle per core (ggml's kernels; two- and four-row
   interleaving and SMT threads measured slower). Levers left: fewer misses (more VRAM for experts: `BNK_DS_PF_CHUNKS=2`
   gives back ~0.6 GiB per GPU when prompts are short), stream a missed expert over PCIe for the GPU while the CPU does
   the others (PCIe 0.75 ms per expert vs CPU 0.29 ms per expert: worth ~20% of the CPU part at 3 misses per layer).
4. **Groups and VRAM**: `BNK_DS_PF_CHUNKS` 4 (8,192-token groups) is the default; 8 halves the per-token PCIe cost of
   long prompts again for another ~2.4 GiB of VRAM per GPU. Measure on the user's prompts.
5. **Quality A/B for the user**: IQ2_XS vs IQ2_XXS on their own prompts (`./run.sh ds-xxs --port 8081`).
6. **Loose ends**: the indexer's `weights_proj` scale (1/64) is not applied (ranking-invariant); `--slots 2` untested;
   `forward_pipe` untested; vision not ported; the Engram gather could run ahead of the group (it is 0.4 s per 2,048
   tokens on stage 0, from NVMe page faults).

## Commands

```bash
cd /opt/engines/bnk-deepseek      # branch ds-perf
cmake --build build --target bnk test_ds_ops test_ds_gemv test_ds_cpu bench_ds_cpu bench_ds_attn -j 24
CUDA_VISIBLE_DEVICES=1 ./build/test_ds_ops && CUDA_VISIBLE_DEVICES=1 ./build/test_ds_gemv
./run.sh ds-xs --port 8081        # production (master, 8080) must not be running: both GPUs are needed
# copy test: a text and its first 512 tokens again, the prefix through the prompt path, the copy scored at 8-token windows
BNK_NLL_T=8 BNK_NLL_SKIP=4096 ./build/bnk nll --model "$M" --tokens-file tok_long4096_copy512.txt --ctx 16384 --threads 24 --gpus 0,1 --counts ~/.cache/bnk/counts-DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS.bnkc
BNK_DS_PF_PROF=1 ./build/bnk run --model "$M" --tokens-file tok_long8192.txt --max-new 32 --ctx 16384 --threads 24 --gpus 0,1 --counts ...
```

Stop only processes you started, by PID (`pgrep -af 'bnk run'`, `kill PID`); a `pkill -f` pattern matches the shell
running the command itself.
