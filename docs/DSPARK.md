# DSpark: DeepSeek-V4.1-Flash's draft head as the DeepSeek engine's drafter (branch `dspark`, started 2026-10-26)

Why now: `docs/DEEPSEEK-NEXT.md` measured what a verify window costs on this machine. With the static expert placement
a verified row cost 0.43 of a plain round (the extra token's CPU expert misses), which left a drafter 10-20% at best;
with the adaptive exchange (v0.6.0) a row costs 0.13-0.19, and the same model puts DSpark at x1.35 (70% acceptance)
to x1.9 (85%) on code. This file is the plan and the record.

## What DSpark is (DeepSeek-V4.1-Flash-1017: `inference/model.py`, `inference/config.json`; the DSpark paper, arXiv
2607.05147)

Three extra DeepSeek-V4.1 blocks (`mtp.0-2` in the checkpoint; `n_mtp_layers` 3) that draft a **block of 5 tokens in
one forward** (`dspark_block_size` 5), semi-autoregressively:

* **Input.** The anchor (the last token the target emitted: the bonus token) and 4 *noise* tokens (`<mask>`, id
  128799) are embedded with the target's token table and expanded to the 4 hyper-connection copies, like any
  position of the backbone.
* **Context.** The target's hidden states at the *attention input* of layers 37, 38, 39 (`dspark_target_layer_ids`;
  each the mean over the 4 hc copies), concatenated (15,360) and projected by `main_proj` [15360 -> 5120] plus
  `main_norm`: one vector `main_x` per verified position. Each stage's attention turns `main_x` into one K/V latent
  (`kv_norm(wkv(main_x))`, RoPE on the last 64 dims at the position) kept in a **ring of the last 128 positions**;
  the block's 5 rows attend to those 128 latents and to each other (all 5, not causal), with the per-head sinks. The
  block's own K/V is not kept. Positions: the ring's are the target's; the block's are `pos .. pos + 4`.
* **Stages.** Each stage is a full V4.1 block: mHC pre-mix, the window-only attention above (q_a / q_norm / q_b, 64
  heads x 512, grouped low-rank output projection o_a / o_b with 8 groups of rank 1024), post-mix, MoE of **128
  routed experts, top-3** (sqrtsoftplus routing with the bias, `route_scale` 1.5, renormalized; the same 2304-wide
  SwiGLU experts as the backbone, so an expert is the backbone's size, 35.4M parameters) plus one shared expert.
  Only the last stage has `norm`; only the first has `main_proj` / `main_norm`.
* **Head.** The target's output head (shared) on the 5 rows -> logits [5 x 129280]. Then the **Markov head** chains
  left to right: for position i the bias `markov_head.head(markov_head.embed(token_{i-1}))` (rank 256) is added to the
  row's logits, and token_i is sampled (the reference at the request's temperature; llama.cpp's in-graph chain takes
  the argmax). token_0 is the anchor. The **confidence head** `sigmoid(proj([h_i ; embed(token_{i-1})]))` estimates
  each position's acceptance probability; the scheduler keeps drafts while the running product stays above a
  threshold (production budgets 4-6 drafts; llama.cpp: `--spec-draft-p-min`).
* **Sizes** (shards 44-46 of 48, fetched alone by `tools/dspark_fetch.py`, 7.5 GB): experts 13.6B parameters in fp4
  (E2M1 with one E8M0 scale per 32: MXFP4 bit-exact, 7.6 GB; a 128-expert layer is 2.5 GB), dense 0.57B in fp8
  (`main_proj` 79M, per stage ~165M incl. the shared expert), the Markov head 2 x [129280 x 256] bf16, the confidence
  projection [5376].
* **Training** (V4.1 paper 2.4.3): after pre-training with the backbone frozen, then alongside post-training with no
  gradient into the backbone. No MTP loss in the backbone's pre-training.

## The llama.cpp precedent (upstream, 2026-10: `conversion/deepseek41.py`, `src/models/dflash.cpp`,
`common/speculative.cpp`, `docs/speculative.md`)

* `convert_hf_to_gguf.py --dspark` exports `mtp.*` as a **DFLASH-architecture GGUF**: stages `blk.0-2` with the V4.1
  tensor names, experts MXFP4 bit-exact, dense BF16, `fc` = main_proj, `enc.output_norm` = main_norm, `output_norm` =
  mtp.2.norm, `markov_w1` / `markov_w2`, `conf_proj` (F32); metadata `dflash.block_size` 5, `target_layers`
  [37, 38, 39], `dflash.hyper_connection.single_pass` true, `sample_from_anchor` true, the mask token id, sliding
  window 128 (+1 for the mask row). The token table and lm head are the target's at runtime.
* Runtime (`draft-dspark`): one decode of anchor + 4 masks per draft round, the Markov chain greedy in-graph, drafts
  truncated below the confidence threshold, `n_max` 5 drafts. Reported on V4-Flash in DeepSeek's production: +51%
  throughput over MTP-1 with experts in HBM.

We use the converter as-is for the GGUF (a synthetic safetensors shard of the fetched `mtp.*` tensors + the repo's
config and index; the venv at `~/.cache/bnk/convert-venv` with upstream `gguf-py`) and read the DFLASH layout in our
loader. Nothing of llama.cpp's runtime is used.

## The port

`DsSpark : Drafter` (`src/deepseek/ds_spark.{h,cpp}`), owned by the head stage of `DsEngine`, replacing the
`LookupOnlyDrafter` (the prompt-lookup drafts keep working through the generator exactly as now: `choose_drafts`).

1. **Weights.** The DFLASH GGUF is loaded as three `DsLayer`s with `n_expert` 128 / `n_used` 3 (`DsModel` gains a
   per-layer expert count; the routing kernel takes `n_expert` and `k` already), `main_proj` / `main_norm`, the
   Markov and confidence heads. Dense weights to the head stage's GPU (0.6 GB as Q8_0/F16, the loader's rule).
   Routed experts **in RAM through the CPU pool first** (MXFP4: `ggml_get_type_traits_cpu` has its vec_dot; the GPU
   R-format does not, so no VRAM residency in step one). A draft forward reads up to 45 experts (5 rows x 3 x 3
   stages) at ~0.2 ms each on the pool: ~9 ms plus the dense GEMVs (~2 ms). Step two: an MXFP4 R-layout
   (`kernels/rfmt`, next to NVFP4's) so the drafter's layers join the expert cache and the exchange as layers 40-42.
2. **Context.** The backbone captures the mean over hc copies of the residual at the attention input of layers 37-39
   for every window row (`DsEngine::layer_forward`: a small kernel into a [kMaxWindow][15360] buffer, pinned to the
   head stage; with two stages the rows cross to GPU 0 with the logits). `Drafter::run(R_rows, next_tokens, n, cell0)`
   receives the verified rows' context (the generator already passes the kept rows and their successor tokens):
   `main_proj` + `main_norm` on the n rows, then per stage `kv_norm(wkv(.))` + RoPE into the stage's ring at
   positions cell0..cell0+n-1 (per conversation slot: 3 stages x 128 x 512 fp16 = 384 KB a slot; `restart` on
   rollback like `DenseDFlash::restart`).
3. **The block.** One forward of 5 rows through the three stages: embed (anchor + 4 masks, `tok_embd` of the target),
   hc expand, per stage: hc pre-mix, attention over the ring (128 latents) and the block (5 rows, full visibility;
   `ds_attention` with a window variant that excludes nothing within the block and ignores the block's positions in
   the ring), o_a / o_b, post-mix, routing + 3 experts a row on the CPU pool (the mailbox path of `moe()`, eager: the
   drafter runs between the target's forwards, nothing to overlap with yet), shared expert, hc post. Then
   `output_norm`, the target's head on 5 rows (one batched GEMM over the Q8_0 head, ~0.7 ms), and the Markov chain on
   the host over the top-k of each row: logits_i += W2 (W1[token_{i-1}]) for the candidates, sample at the request's
   temperature (the generator verifies sampled drafts by rejection sampling against the drafter's distribution:
   `draft_dist`, exactly as DFlash2's), confidence_i for the schedule.
4. **The interface.** `run(..., draft=true)` returns draft 0 and keeps the block; `step()` returns drafts 1-4 (as
   `DenseDFlash` does) and stops at the first position whose running confidence product falls under a threshold
   (`BNK_DS_SPARK_PMIN`, default 0.5: measured against the verify cost, 0.13-0.19 of a round per draft row, a draft
   pays above ~0.2 acceptance; the threshold trades the window's cost against the expected tokens like the paper's
   scheduler, with our own round-time curve from the oracle sweep). `max_draft` for the engine becomes 5.
5. **Graphs.** The block forward is one CUDA graph per stage count (5 rows fixed); the CPU experts are the eager
   mailbox exchange as in `moe()`'s eager path. The ring update for n rows runs on the compute stream after the
   target's forward.

Validation, in order: (a) the converter's GGUF loads and its stage-0 attention / MoE on a fixed input matches a
NumPy reference of `inference/model.py`'s math (a `tests/test_ds_spark.cpp` with dumped activations from a small
Python run over the fp8/fp4 weights: no torch needed for one block of 5 rows); (b) `bnk run` with the drafter on a
code prompt: `MTP a/b in c rounds` acceptance per position, drafts a round, tokens a round; (c) the nll/KL check
against the plain path (`BNK_NLL_REF`, the output distribution must not move: sampled drafts with rejection sampling
keep it exact, greedy is unchanged by construction); (d) the oracle sweep's round-time curve against the measured
acceptance decides the confidence threshold; (e) the agent session.

Open questions, to settle while building: whether the reference keeps the block's own K/V out of the ring (the
Python does: `window_kv_cache` takes `main_kv` only; llama.cpp's "+1 for the mask row" is its masking trick); the
exact hc pre-mix state carried between the three stages (`make_identity_pre_mix` at the block's start, as the
backbone's start); the Markov chain's sampling vs argmax at temperature 1.0 (sampled drafts keep the target's
distribution and the paper reports its lengths at temperature 1.0); where the 7.6 GB of MXFP4 experts sit in RAM
next to the 99 GB arena (pinned, their own store).

## Status

* 2026-10-26: branch and worktree (`/opt/engines/bnk-dspark`). `tools/dspark_fetch.py` fetched the 2,401 `mtp.*`
  tensors (7.93 GB) into `~/.cache/bnk/dspark`; `tools/dspark_gguf.py` wrote them as one safetensors shard and ran
  upstream llama.cpp's `convert_hf_to_gguf.py --dspark` (checkout d854d98, the venv at `~/.cache/bnk/convert-venv`):
  `~/.cache/bnk/dspark/DeepSeek-V4.1-Flash-1017-DSpark.gguf`, 8.5 GB, 78 tensors (experts MXFP4 [5120 x 2304 x
  128] a matrix, dense BF16, norms / router / bias / sinks / mixers F32, `conf_proj` F32 [5376]), metadata as in the
  precedent section (`dflash.sliding_window` 128 as is: the +1 is llama.cpp's runtime). `src/deepseek/ds_spark.{h,cpp}`:
  the `DsSpark` drafter's interface and its loader (the stages as `DsLayer`s, the experts into their own pinned
  `ExpertStore`, run on the engine's pool through `CpuExpertPool::run(model, store, ...)`), wired into `DsEngine`:
  `--mtp FILE` loads it on the last stage (the head's and the target layers'), `ds_hc_mean` captures the window rows
  at layers 37-39 into its target buffer inside the forward graph, slot select / reset / rollback follow into its
  rings, `mtp()` returns it. Builds. The forward (`feed`, `draft_block`: steps 2-4 of the plan) is not written: the
  two stubs throw. Not yet run: the loader needs the backbone loaded, i.e. the GPUs free of the server.
  Next: run `build/bnk run --mtp ~/.cache/bnk/dspark/DeepSeek-V4.1-Flash-1017-DSpark.gguf ...` for the load check
  (store build time and RAM: 7.2 GB pinned next to the 99 GB arena), then `feed` (fc + norm + per-stage wkv/kv_norm/rope
  into the rings), the block forward, the heads.
