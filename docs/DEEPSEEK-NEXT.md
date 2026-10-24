# DeepSeek-V4.1-Flash: where things stand and what comes next (handoff, 2026-10-24 04:30)

Read this first when resuming. `docs/DEEPSEEK.md` describes the model, the memory plan and the code; this file is
the state and the plan.

## Where things stand

* Branch `deepseek`, worktree `/opt/engines/bnk-deepseek` (master in `/opt/engines/bnk` is untouched and stays
  production). **Nothing is committed yet**: `git status` lists the whole port. Commit it first thing
  (`git add -A && git commit`), the user's go-ahead was pending.
* Serves correctly: `./run.sh ds-xs --port 8081` (IQ2_XS, 155 GB) or `./run.sh ds-xxs --port 8081` (IQ2_XXS,
  138 GB, the build ISTA recommends). Both backbones and the shared Engram file (110.6 GB) are in the Hugging Face
  cache under their snapshot commits (the configs pin them); `tools/models.py resolve ds-xs` prints the path.
  The worktree's `.venv` and `serve/web/dist` are symlinks into the main repo.
* Measured on the two V100s (IQ2_XS, 1 slot, context 262144, routing counts from a few sessions):

  | | |
  |---|---|
  | Decode | 12.6-13.8 tok/s (plain, no drafter) |
  | Prompt reading | 100-118 tok/s (batched chunks of up to 2,048 tokens), ~20 through decode windows |
  | Prose perplexity (336 tokens) | 2.88 at 1-token windows, 2.90 at 8 |
  | Copy test (second copy of a 336-token passage) | ppl 1.002, top-1 100%: long-range attention is right |
  | Load | ~4 min (caches filled from the file, then ~90 GiB copied into pinned RAM) |
  | Placement | ~4,900 experts VRAM-only (22.8 GiB of cache per GPU), ~10,400 in RAM, Engram on NVMe |

  Strata 0.2.4 (which added this model the same night, see `docs/DEEPSEEK.md`) reports 5-11 tok/s on single
  12 GB cards and 31 on four R9700s; ISTA's card says IQ2_XXS keeps 100% of the base model's task average and
  Strata measured IQ2_XS at ppl +7.3% / IQ2_XXS +15.5% against the original.
* Tool calls: the model writes `<｜DSML｜ calls>` / `<｜DSML｜ invoke ...>` (not the template's `tool_calls`);
  `serve/chat.py` accepts both spellings and the server returns proper `tool_calls` (tested in a round trip).
  Thinking on/off and `reasoning_effort` (1-100, low/high/max) go through the GGUF's template.
* Tests: `build/test_ds_ops` (every new kernel vs a CPU reference), `build/test_ds_gemv` (GEMV paths at 1-8
  tokens), `build/test_ds_cpu MODEL [layer]` (the CPU pool on real experts, plus a mailbox replay mode),
  `V=129280 build/test_topk`. Diagnostics in the engine: `BNK_DS_TRACE=0,1` (per-stage rms in eager mode),
  `BNK_DS_TRACE_DUMP=prefix` (layer 0's mailbox), `bnk hdump` with `BNK_HDUMP_T` / `BNK_HDUMP_T2` (per-layer
  residual dumps at two window sizes in one load; `scratch/cmp_hd.py` compared them).
* The one bug that cost the night: the engine never called `ggml_cpu_init()`, so ggml's single-row `vec_dot`
  (the CPU pool's path for an expert with one token) returned zeros. Any new engine or tool that runs
  `CpuExpertPool` must call it (memory note `ggml-cpu-init-new-engines`).

## Knobs

* `BNK_DS_STAGE_MIB` (default 1024): the prompt path's expert staging arena per buffer, two buffers per GPU. Bigger
  = faster prompts (more experts streamed over PCIe instead of computed on the CPU), smaller expert cache = slower
  decode. A whole layer's outside experts would be ~6 GiB.
* `--prefill-chunk` (default 2048; `--prefill-chunk 0` = decode windows only): the prompt buffers take ~1.5 GiB at
  2,048 (q and attention outputs are 2048 x 64 x 512 floats each). 1,024 saves ~0.8 GiB of cache per GPU.
* `--cache-gib A,B`: cap the expert caches per GPU. `--slots N`: conversations decoded together (untested beyond 1
  for this engine; the code paths are the Qwen engine's).
* `BNK_GEMV_MMA_MIN=1`: tensor-core GEMVs for 1-2 token windows too (int8 activations otherwise). On the prose
  test it made no measurable difference once the real bug was fixed; keep the default.

## Next steps, in order

1. **Commit the branch** (10 min). Then run the kernel tests once more and `./run.sh ds-xs` as the smoke test.

2. **Prompt path speed** (half a day to a day; target 250-400 tok/s on long prompts). It is PCIe-bound: a long
   chunk uses nearly every expert of a layer, and the non-resident ones stream from RAM at 13 GB/s (~3.6 GB per
   layer for IQ2_XS). Three independent gains:
   * **Overlap the two GPUs**: today a chunk goes through stage 0, then stage 1 (`DsEngine::prefill_chunk` runs
     the stages one after another on the caller's thread). Run chunk k+1 on GPU 0 while GPU 1 does chunk k, as
     `Engine::prefill_chunk` does with its `Worker` thread and two transfer buffers (`pf_xfer_in_[2]`). Needs a
     second `pf_xfer_` buffer and the CPU pool shared between the two (it takes one caller at a time: fine).
     Expected: ~2x on prompts of two chunks or more.
   * **Prefetch the next layer's experts** during the current layer's attention: `Engine::pf_stage_layer` /
     `prefetch_min` stream *every* non-resident expert one layer ahead for chunks of 1,024+ tokens; with the
     1 GiB staging cap that means the most-used ~70 experts (counts from the layer's routing are only known after
     its router runs, so prefetch by the ranking, or raise the cap for long reads and give the VRAM back after,
     as `prefill_staging_shrink` does).
   * **The CPU share**: `opt_.cpu_us_per_expert_token` (46 us, measured on 2.2 MB Qwen experts) is scaled by blob
     size in `pf_moe`; measure it on this model's 15 MB experts (`BNK_SVC_PROF`-style timing around `cpu_->run`)
     and retune `BNK_PCIE_US_PER_MB` / `BNK_CPU_US_PER_EXPERT_TOKEN`.
   Also: the indexer scores for a chunk at long contexts (`score_rows` x `lds_`) and the per-row radix select are
   untested beyond 16K of context; try a 60K-token prompt once and watch `ds_index_scores` time.

3. **DSpark drafter** (a day). The model's own 3-layer MoE draft head (128 experts, block size 5, with a Markov
   head and a confidence head: `inference/model.py` DSparkBlock in the scratch copy of the official repo; the
   DeepSeek GGUF converter in llama.cpp PR 30817 writes it as a DFLASH GGUF, Unsloth publishes
   `DeepSeek-V4.1-Flash-1017-DSpark` Q8_0 at 8 GB). Strata measures +30-50% decode with it. Fit: 7-8 GB of VRAM
   (its experts), i.e. ~500 fewer resident experts per card, so measure the net. The engine's `Drafter` interface
   (`engine_api.h`) and the dense engine's DFlash2 (`src/dense/dense_dflash.cpp`) are the templates; the drafter
   reads the attention inputs of layers 37-39 (`dspark_target_layer_ids`), so the last stage must keep those.
   Without it, prompt-lookup drafting is also off (the generator only drafts when a drafter exists:
   `Generator::draft_from`); enabling lookup drafts alone (verify windows of up to 8 rows) is a cheap first step
   for code edits and quoted text and exercises `commit()` with partial windows, which no test has yet.

4. **Placement and decode speed** (ongoing). Decode is bound by the CPU experts (~63% of routed experts missed
   the caches on the first run; with counts the resident set follows the traffic). Ideas: shrink the prompt
   buffers (`--prefill-chunk 1024`) and the staging (`BNK_DS_STAGE_MIB 512`) when decode matters more; seed the
   ranking from Strata's `data/dsv41/profile-2.25.bin` (format unknown, 66,048 bytes, probably counts per
   (layer, expert)); cold experts in a smaller format (`docs/EXPERT-MIX.md`) do not apply (the file's formats are
   already 2-bit). The CPU pool runs ggml's AVX2 `vec_dot` for single-token experts: Strata's AVX-512 kernels
   do not help on this EPYC (AVX2 only), but `mdot` for n = 1 might; profile first.

5. **Quality A/B for the user**: IQ2_XS vs IQ2_XXS on the user's own prompts (`./run.sh ds-xxs --port 8081`).
   Both configs sample at temperature 1.0 / top_p 1.0 as the model card says. If the user prefers IQ2_XXS, it is
   17 GB smaller and ISTA's measured task average is the base model's.

6. **Loose ends** (each small): the indexer's `weights_proj` scale (1/64) is not applied (ranking-invariant, so
   harmless, but apply it for fidelity); multi-slot serving untested (`--slots 2`); `forward_pipe` (two lanes on the
   pipeline) is implemented but untested; vision (`DeepSeek-V4.1-Flash` has a ViT, llama.cpp's PR converts an
   mmproj) not ported; the GEMV test's F16 tolerance line is a comment-only change; `tests/test_ds_cpu.cpp` and the
   trace hooks can stay.

## Commands

```bash
cd /opt/engines/bnk-deepseek
cmake --build build --target bnk test_ds_ops test_ds_gemv test_ds_cpu -j 24
CUDA_VISIBLE_DEVICES=1 ./build/test_ds_ops && CUDA_VISIBLE_DEVICES=1 ./build/test_ds_gemv
./run.sh ds-xs --port 8081            # or ds-xxs; production (master, 8080) must not be running: both GPUs are needed
# perplexity / copy test (model path from tools/models.py resolve ds-xs; token files in the session's scratch dir
# or make new ones with serve.tokenizer):
BNK_NLL_T=1 ./build/bnk nll --model "$M" --tokens-file tok.txt --ctx 16384 --threads 24 --gpus 0,1 --counts ~/.cache/bnk/counts-DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS.bnkc
BNK_NLL_T=1 BNK_NLL_SKIP=336 ./build/bnk nll --model "$M" --tokens-file tok_doubled.txt ...   # second copy: expect ppl ~1.00
```

Stop only processes you started, by PID (`pgrep -af 'bnk serve'`, `kill PID`); a `pkill -f` pattern once matched
the shell running the test itself.
