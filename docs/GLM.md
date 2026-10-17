# GLM-5.3-Flash on the V100s (branch `glm-flash`, started 2026-10-17)

Goal: serve NVIDIA's `nvidia/GLM-5.3-Flash-FP4-Local` (ModelOpt NVFP4 experts, quantization-aware distilled from
`zai-org/GLM-5.3-Flash`: GPQA 91.3 against 92.1 for BF16) on the two V100s with the same approach as the Qwen
FP4-Local port (`docs/NVFP4.md`): every weight the checkpoint's own, the formats decoded in bnk's existing
quantized kernels. Then find what this model's shape allows.

## The model

`Glm5NextForConditionalGeneration` (transformers 5.16 `modeling_glm5_next.py`), text part only:

| | |
|---|---|
| Size | 320B parameters, 18B active; 45 layers, hidden 4096, vocab 154,880, context 1M (NoPE: no position encoding at all) |
| Attention | 34 **KDA** layers (Kimi delta attention: linear attention with a decay per key channel), 11 **MLA + DSA** layers (every 4th from layer 3: DeepSeek-style latent attention, q rank 1536, latent 512, 64 heads of 256 (K and V), with the DeepSeek sparse-attention indexer compressed in pools of 4 tokens, top 2048 tokens per query) |
| MLP | layers 0-2 dense (12,288 wide); layers 3-44 **288 routed experts** (2,048 wide, top-8, sigmoid scores + correction bias, weights normalized and scaled by 2.5) plus one shared expert |
| Residual | **mHC** hyper-connections: 4 residual streams of 4096; each block collapses them with a learned `pre` weight, expands its output with `post` and mixes the streams with a 4x4 doubly-stochastic `comb` (20 Sinkhorn iterations), all three predicted per token from the flattened normed streams |
| Activation | SwiGLU with a clamp (`swiglu_limit` 10: gate capped, up clamped to +-10) in the experts, the shared expert and the dense MLPs |
| Extras | an MTP layer (layer 45: a full MLA + MoE layer over the final residual and the next token's embedding) for speculation; a vision tower. Neither is used yet |

Per layer (bnk's tensor names, `tools/convert_glm_fp4.py`):

* `hc_attn_fn` / `hc_ffn_fn` [24, 16384] (BF16), `*_base` [24], `*_scale` [3]: the mHC projection of the flattened
  streams (unweighted RMS norm) into pre[4] | post[4] | comb[16]. `attn_norm`, `ffn_norm`: the block input norms.
* KDA: `kda_qkv` [24576, 4096] (q | k | v, each 64 heads x 128, from MXFP8 -> Q8_0), `kda_conv` [24576, 4] (depthwise
  causal conv, silu), `kda_fgb_a` [320, 4096] (f_a | g_a | b, BF16), `kda_f_b` / `kda_g_b` [8192, 128]: the forget gate
  `g = -5 sigmoid(exp(A_log[h]) (f_b(f_a(x)) + dt_bias))` per channel and the output gate `sigmoid(g_b(g_a(x)))`;
  beta = sigmoid(b x) per head; `kda_o_norm` (RMS norm per head, gated), `kda_out` [4096, 8192]. The recurrence per
  head: `S = diag(exp(g)) S; S += k (v - S^T k)^T beta; o = S^T q / sqrt(128)` with q, k l2-normalized.
* MLA: `attn_qkv_a` [2048, 4096] (q_a 1536 | kv_a 512), their RMS norms, `attn_q_b_idx` [20480, 1536] (q_b: 64 x 256 |
  the indexer's wq_b: 32 x 128), `attn_kv_b` [32768, 512] (per head 256 K rows then 256 V rows, BF16, kept F16),
  `attn_output` [4096, 16384]; the indexer's `idx_x` [288, 4096] (wk 128 | weights_proj 32 | pool gate 128),
  `idx_k_norm` (a LayerNorm with bias), `idx_ape` [4, 128].
* MoE: `ffn_gate_inp` [288, 4096] (BF16) + `.bias`, `ffn_gate_up_shexp` [4096, 4096] + `ffn_down_shexp`,
  `ffn_{gate,up,down}_exps` NVFP4 [288, 2048, 4096] / [288, 4096, 2048] with `.scale` [288] (the fp32 globals).
  Dense layers: `ffn_gate_up` [24576, 4096], `ffn_down`.

## The checkpoint and its conversion

177 GB in 40 shards plus `tensors/native-QB.safetensors` (the KDA gate projections, the indexers, kv_b) and
`model-inputscales.safetensors` (activation scales, unused). Formats: routed experts NVFP4 (e2m1 codes, one e4m3
scale per 16, fp32 `weight_scale_2`), the trunk's shared experts and MLA projections FP8_PB_WO (e4m3 with one fp32
scale per 128 x 128 block), the KDA / dense-layer / lm_head projections MXFP8 (e4m3, one UE8M0 exponent per 32),
the rest BF16 / F32.

**`csf_32x2_v1`** (the experts' block scales, "lossless compression", no public decoder): a matrix's [rows,
cols/16] e4m3 scale bytes in 32 pages of rows/32 rows; per row a u8 base (`csf32_bases`), per scale a 4-bit code
(`csf32_codes`, low nibble first): byte = base + code. Scales the code cannot reach are exceptions:
`csf32_page_bases` [32] u32 holds each page's **byte** offset into `csf32_exceptions` (u16), `csf32_offsets`
[32 x 257] u16 the exception index relative to the page at the start of each of the page's 256 segments of 64
scales (and its end); an exception is `(byte << 8) | position in the segment`. Verified on real experts: the
decoded scales reach exactly 6 x 448 at the block holding the matrix's amax (every block's largest code is 6, so
the quantizer normalized per block), no NaN bytes, and every exception sits where the code saturated (0 or 15).

`tools/convert_glm_fp4.py` plans the whole GGUF first (every tensor's offset; a sparse file of the final size
and `OUT.plan.json` beside it), then downloads one shard at a time (~85 MB/s here), converts it and writes in
place (`pwrite`), and deletes it, so the source never sits on disk; it resumes (`--resume`) and can fill the
experts of some layers now and others later (`--expert-layers A-B`). The whole model converted in 25 minutes
(dense 2 minutes). Output: 181.7 GB (experts 171.2 GB = 159.5 GiB, dense 10.4 GB), ggml-compatible tensors (the
NVFP4 blocks are llama.cpp's layout, `tests/test_dequant` and `test_moe_tc` work on them). The MXFP8 / FP8_PB_WO
projections are stored as Q8_0 (the one re-quantization, as for Qwen). The tokenizer (byte-level BPE, the GLM-4
pre-tokenizer: `tokenizer.ggml.pre = glm4`), the chat template (it uses `{% break %}`: Jinja's loopcontrols) and
the three end-of-turn ids (`glm5next.eos_token_ids`) ride along.

## Where the weights live

159.5 GiB of experts cannot be in RAM (128 GB) and VRAM (2 x 32 GB) at once, so the expert store has three tiers:

1. the VRAM caches (one per GPU, the hottest experts of its layers, R layout, as before: ~16 GiB each after the
   dense weights, the context and the prompt buffers);
2. a page-locked host arena, capped (`--ram-gib`, default 100 GiB for this model): the ranking's next experts
   after the ones the GPUs hold (filled after every stage's cache is up; `ExpertStore::build` skips GPU-resident
   experts), DMA'd into the caches and the prompt stage, computed in place by the CPU pool;
3. the model file's mapping for the rest (the page cache, then the NVMe at 2.9-3.2 GB/s sequential): three
   separate matrices per expert (`ExpertStore::parts`), read by the CPU pool and copied to the GPU in three pieces.

The VRAM also holds: the dense weights (4.4 + 4.2 GiB over the two stages, layers 0-23 / 24-44), the latent KV
cache (512 fp16 per token per MLA layer: 1 KiB, plus the indexer's keys and gates 512 B and the pooled keys 128 B
per token), the KDA state (64 x 128 x 128 fp32 = 4 MiB per layer per conversation), the prompt buffers.

## The engine

The forward driver (windows, CUDA graphs, batching, the two-GPU pipeline, the expert mailboxes, parking) is
shared; `Config::is_glm()` switches the layer (`src/engine/engine_glm.cpp`) and the loader
(`src/core/model_glm.cpp`). The Qwen fields double for what the shared code sizes: the KDA conv and state are
DeltaNet's (`ssm_*`, conv channels q | k | v), the latent cache is "one KV head of 512" (`n_head_kv = 1`,
`head_dim = kv_lora`, no V cache), the DSA indexer is QSA's shape (`idx_*`, `compress_ratio = kpool`): the same
per-conversation buffers, parking, snapshots and VRAM elasticity apply.

New kernels (`src/kernels/glm_ops.cu`): the mHC norm / weights (Sinkhorn per token) / collapse / combine;
the KDA gates and the delta rule with a decay per key channel (`kda_recurrence`, from `gdn_rec4_k`); the MLA
latent store, the absorbed queries (`q_abs = W_UK^T q` per head, 512 wide), split-KV decoding and one-warp-per-row
prompt attention over the latent (K = V = the cached row), the value expansion `W_UV o_lat`; the DSA indexer:
LayerNorm'd keys and pool gates stored per cell, the pooled keys (a learned softmax over the 4 cells, per dim),
the scores `sum_h w_h relu(q_h . pool / sqrt(128)) / sqrt(32)` and QSA's radix top-k over the complete pools
before the query's tail (prompts within 2,051 tokens are dense, as in the model); sigmoid routing with the bias;
the SwiGLU clamp in every expert path (GPU hits, list and tensor-core prompt kernels, the CPU pool).

Validated against `tools/glm_ref.py` (an fp32 CPU forward of the HF modeling code on the same GGUF, the routed
experts loaded on demand) on "The capital of France is": layer 0 (KDA, dense MLP, mHC) agrees to 1e-4 relative
error in every intermediate, layer 3 (MLA, MoE) to 3e-4 for the attention and 6e-3 for the MoE output (the int8
activation path, as for Qwen); greedy decoding gives " Paris. France is" and the reference's top token is " Paris".

## Validation of the prompt path (2026-10-17, later)

`tools/glm_cmp.py` compares the reference's dumps with bnk's prompt-path dumps row by row. A 2,000-token prompt
(one chunk, dense attention, two attention pieces) agrees at layer 3 to 2-3e-4 (attention) and 8e-4 (MoE: a few
rows route differently at near-ties); a 2,500-token prompt (the DSA selection active for rows past 2,051, the
reference implementing the k-pool indexer) agrees the same way, the worst sparse row at 7e-3. Two bugs found on
the way: the dense MLP's 1,024-row pieces reused the GEMM's fp16 conversion buffer that held the next piece's
input (rows past 1,024 were garbage), and the indexer scores (signed: the head weights have either sign) were
compared as unsigned by the radix top-k.

## Performance work (2026-10-17, evening)

Measured with `bnk run` on the two V100s, a 90 GiB arena, the routing counts of earlier runs as the ranking
(`--profile`), a 2,784-token README prompt (`readme2k`) and four copies of it (11,136 tokens).

| Step | Decode (tok/s) | 2.8K prompt | 11K prompt |
|---|---|---|---|
| Uniform ranking, every miss through the file mapping | 0.36 | | |
| Experts outside the arena `pread` into a pinned stage / scratch | 3.0 | 57 tok/s | |
| Accumulated routing counts as the ranking (38% misses) | 10.3 | 60 | |
| The prompt path stages the next layer on its own thread | 11.5 | 61 | |
| File-tier experts read in slices by every pool thread, 16 disk threads | **14.2** | 58 | 160 (3 x 3,712) |
| Double-buffered disk stage, the biggest layout from the first chunk | | 68 (1 chunk) | 166 (2 x 5,568) |

What was learned:

* **The CPU pool's cost was the file tier.** The per-layer profile (`BNK_SVC_PROF=1`, now with the file-read
  count) showed a layer whose misses are all in the arena costs the CPU 0.7 ms (2.6 experts: 0.27 ms per 14 MB
  expert, 24 threads), while each expert read from the model file cost 7.8 ms: 12% of the layer-forwards, 60% of
  the CPU time. One thread read each expert; the NVMe gives 2.0 GB/s to one reader, 3.3 to four and 5.3 GB/s to
  sixteen (measured with `dd iflag=direct`), so `ExpertStore::read_blob_part` now splits an expert's three ranges
  across every thread of the pool (3.5 ms per file-tier expert) and the prompt path's disk stage uses 16 threads.
* **GPUs reading missed experts over PCIe lose.** The hit kernels can read an arena expert straight from the
  mapped page-locked memory (`--host-share`, `MoeLayerDesc::host_of`, the ggml-layout `moe_gu_k<.., HOST>`): it
  works (the Paris prompt), but a 14 MB expert costs the GPU ~1 ms on PCIe against the CPU pool's 0.3 ms, and the
  GPU time is on the critical path of its stage: a 0.5 share gave 6.4 tok/s against 11.5. Off by default, kept
  as an option.
* **The prompt path is expert staging, not compute.** Every non-resident expert of a layer (about 238 of 288,
  3.3 GB) is copied to the GPU per chunk per stage: ~2.3 GB from the arena by DMA (12 GB/s) and ~1.1 GB from the
  file (the NVMe, shared by both stages). `BNK_PF_PROF=1` now prints the wait for the staging: ~640 ms of a
  700 ms MoE layer at 3,712 tokens, the GPU's own work ~55 ms. The disk stage is double-buffered (a wave's reads
  overlap the previous wave's copies: 816 -> 694 ms per layer). Since a chunk's cost is fixed whatever its size,
  glm5next takes the biggest prompt layout that fits from the first chunk (`prefill_chunk_max` auto: 8,192; a
  prompt that fits one chunk gets a layout its own size, borrowing no more VRAM than needed), and the stages
  regrow their caches concurrently after the prompt (the borrowed VRAM is refilled from the model file: the hottest
  experts are not in the arena by design).
* A dense layer has no expert slots: the cache's regrow skipped that and threw after a long prompt (fixed).

Where it stands: decode is bound by the CPU pool's ~40 ms per token for the ~50-85 missed experts (plus the GPU
stages' ~30 ms); the forward's floor without misses would be ~35 ms (28 tok/s). Prompts are bound by the model
file and the arena streaming through the GPUs: ~18 s per chunk per stage, so long prompts read at ~170-300 tok/s
and the arena size (`--ram-gib`: 100 GiB in the config, 90 in these runs) trims both.

## Status and next steps

* 2026-10-17: converts, loads on both V100s, decodes correctly (eager and graph mode), the prompt path validated;
  decode 14 tok/s, prompts 68 tok/s (2.8K, one chunk) to ~170 tok/s (11K) with the ranking from the counts file.
* Next: the MTP drafter from layer 45 (expected to pay little while decode is miss-bound: a window of k tokens
  multiplies the misses), a CPU NVFP4 kernel that beats the AVX2 dot (0.27 ms per expert now), and for prompts a
  smaller borrow per chunk (the 8K layout takes 11 GiB) or an incremental regrow; the server end to end
  (`./run.sh glm`, two slots, 8 GiB of parking).
