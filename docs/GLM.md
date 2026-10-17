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

## Status and next steps

* 2026-10-17: converts, loads on both V100s, decodes correctly (eager and graph mode). Speed is not yet
  measured with a routing ranking: a uniform ranking leaves ~80% of expert uses off the GPU and most of those in
  the file-mapped tier.
* Next: a routing ranking from real prompts (the counts file), the exclusive RAM tier, then measurements
  (decode, prompt reads at 4K-32K), then the MTP drafter from layer 45, the server's GLM tool-call parser
  (`<tool_call>name<arg_key>..`), thinking handling (`<think>` opens every turn), and the prompt-path kernels for
  long contexts (the indexer score kernel is the per-row one; the absorbed products read kv_b per row).
