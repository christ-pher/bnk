# DeepSeek-V4.1-Flash: the October 25-27 repository updates

`ds-xs` and `ds-xs-abl` are pinned to ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF at `85caaf74` (Oct 23). The
repository changed after that; the weights did not. Checked 2026-10-28: shards 2-4 of the 4-shard layout have the
same SHA-256 at every later revision; shard 1 at `653a9874` differs only in `tokenizer.chat_template` (37 tensors
sampled). In the new 6-shard layout every tensor has the same type and size as ours, and 64 KB samples at the start,
middle and end of each one (1,042 tensors, plus 2,002 samples of the Engram table at unchanged offsets) match our bytes,
256 MB compared in all. That is a sampled check, not a full download. This plan takes what is
worth having: the vision path (the model is `image-text-to-text`: the base checkpoint has a 32-layer ViT, an aligner
and a separate router bias for image tokens) and the chat-template fixes, and leaves the file layout alone.

## What changed upstream

| Date | Commit | Change | For us |
|---|---|---|---|
| Oct 25 | `41213f3c` | template fixes: V4.1 DSML spelling, tool-result order, namespaces, reasoning history (1,332 prompt comparisons against `encoding.py`) | compare with ours (item 1) |
| Oct 25 | `1c100b97` | Unsloth-based template: image blocks, images in tool results | needed for vision |
| Oct 25 | `e8f84371` | in-place template updater for old downloads | not used (rewrites HF blobs) |
| Oct 25 | `653a9874` | `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf`, 0.97 GB | the encoder (item 2) |
| Oct 27 | `a419d74c` | llama.cpp's upstream layout (b11870+): 6 shards, Engram file as shard 6, `exp_probs_b_vl` embedded, mmproj projector `deepseek41v` | layout: not adopted; take the mmproj and `exp_probs_b_vl.f32` |
| Oct 27 | README | IQ2_XS measured: 99.8% of the base task average (GPQA 90.40, AIME25 / LCB v6 equal) | config note |

## 0. Ground rules: the layout stays

* **Never run `convert_ds41_upstream.py` on our cache.** It rewrites the shared Engram file's header in place (416 bytes,
  "the table data stay"): that is the HF blob `eb19f30d...` behind both ds-xs and ds-xs-abl. Afterwards our loader
  refuses it (`deepseek41-engram` with per-layer `blk.N.engram_embd` becomes one `per_layer_token_embd` table) and the
  production server does not start.
* Why not migrate: the quantized bytes are identical; the loader would need the new names (`engram_kv`, `engram_q*k`
  folded into an F32 `engram_gate`, the hash constants moved from tensors to metadata with absolute row offsets,
  `attn_output_a` as 2D); `tools/tensor_patch.py` and the abliterated folder would follow; the new shards need 121-155
  GB next to the old ones (194 GB free). The only gain would be llama.cpp b11870 as an end-to-end reference, and item
  2's checks do without it (DeepSeek's PyTorch encoder is small enough to run on the CPU).
* The model pin stays at `85caaf74`. The two new files come from `a419d74c` (item 2a).

## 1. Chat template (small, independent of vision)

Today: the GGUF's V4 template (147 lines) with `dsml_v41_template` (serve/chat.py) respelling the DSML tags at load.
It already handles `reasoning_effort` (low/high/max, 1-100). ISTA's current template (303 lines) adds image blocks
(`<｜deepseek_image｜>`, also inside tool results) and rejects the image token in text.

1. Harness: render ours (patched) and ISTA's `chat_template.jinja` against `encoding/encoding.py` of the base repo on
   its five test cases plus prompts from our agent sessions (tools, tool results, multi-turn thinking, effort values,
   mid-conversation system messages). Report every difference.
2. If ISTA's matches `encoding.py` where ours does or better: ship it in the repo (`templates/deepseek41.jinja`) and add a
   config key `"chat_template_file"` that overrides the GGUF's; `dsml_v41_template` then becomes a no-op for these configs
   (keep it for other V4.1 GGUFs). If ours is better somewhere, patch ISTA's.
3. Check: `tools/agent_bench.py` on ds-xs, same tool-call parse rate as now; the server's effort mapping
   (serve/server.py:415) still reaches the template.

## 2. Vision

### 2a. Files and configs

* `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` at `a419d74c` (projector `deepseek41v`): `v.patch_embd` (14x14x3 ->
  1024, bias), 32 x `v.blk.N` (ln1/ln2 RMSNorm, `attn_qkv` + bias, `attn_out` + bias, `ffn_gate`/`ffn_up`/`ffn_down`),
  `v.post_ln`, the aligner `mm.0` (9216 -> 5120) and `mm.2` (5120 -> 5120) with biases, and the three learned
  embeddings `v.token_embd.img_start`, `v.token_embd.img_end`, `v.image_newline` (5120 each). 485M parameters.
* `exp_probs_b_vl.f32` (61,440 bytes = 40 layers x 384 experts, F32; SHA-256 is checked in ISTA's script): the router
  bias for image-span tokens (`gate.bias_vl` of the checkpoint).
* tools/models.py: the mmproj today shares the model's `revision`; add `"mmproj_revision"` and an `"extra"` list of
  `{file, revision}` for the bias, both downloaded by `./models.sh download`. ds-xs-abl takes both through
  `"mmproj_from"`: its patch touches only `attn_output_b` in layers 11-28.

### 2b. Server (serve/vision.py)

DeepSeek's image grid differs from Qwen's (`inference/image_processor.py`):

* patch 14 px, aligner 3x3 -> one LLM cell per 42x42 px; images smaller than 544x544 pixels in area scale up, larger ones
  scale down until the span costs at most 1,024 tokens: `h x (w + 1) + 2` (a start cell, rows of `w` image cells each
  closed by a newline cell, an end cell). A 1920x1080 screenshot comes out at about 1764x966 (991 tokens);
* letterboxed with grey 127 (`ImageOps.pad`), not stretched; normalized (x - 0.5) / 0.5.

Work: split `Images` into a grid policy per projector (Qwen's `smart_resize` / 32, DeepSeek's above); the template's
placeholder expands to the whole span, every cell carrying the image's id (`-1 - hash`) as today, so prefix reuse,
parking, forking and snapshots keep working. The cell's role (start / image / newline / end) follows from its offset and
the grid, so the request's `{id, path, h, w}` is enough. The model's cap (1,024) replaces `--image-max-tokens` for this
projector.

### 2c. Encoder (src/vision)

A second tower beside Qwen's, chosen by `clip.projector_type`:

* ViT: 32 layers x 1024, 16 heads of 64; 2D RoPE (rotary dims split between row and column, theta 1e4) and no position
  table; RMSNorm (eps 1e-6); SwiGLU MLP 2816; final RMSNorm.
* aligner: the patch grid padded to multiples of 3, each 3x3 block unfolded (9216) -> Linear -> GELU (erf) -> Linear ->
  5120.
* Weights as Qwen's: fp16 in page-locked host RAM (~0.97 GB), streamed one layer ahead. The BF16 -> fp16 conversion needs a
  range check first (fp32 for any tensor that overflows).
* **Attention needs the fused flash-attention kernel** (the open item in VISION.md). At the 1,024-token cap an image is up
  to ~9,200 patches; scores through memory would be 9,200² x 16 heads x 2 bytes = 2.7 GB. Fused, the activations are
  ~40 MB per buffer. Qwen's encoder gets the same kernel.
* VRAM: Qwen's encoder borrows the expert cache for an encode; the DeepSeek cache cannot shrink and give it back (the
  2026-10-25 budget crash). Plan: a fixed workspace (~0.3 GB: activations plus the two layer buffers) taken from the expert
  cache budget on GPU 0 at load, with vision configured. Measure the decode cost of the smaller cache (expected small:
  about 1-2 expert slots per layer).

### 2d. Language model (src/deepseek)

1. **Embeddings**: image cells take the encoder's rows, start/end/newline cells the learned vectors, in the prompt
   chunk (ds_prefill.cpp:806 `dequant_gather`). Image cells always go through prompt chunks, never the decode window
   (ds_engine.cpp:950), as in the Qwen engine.
2. **Routing**: image-span tokens select experts with `exp_probs_b_vl` instead of `exp_probs_b`, in all 40 layers
   (the bias only picks experts; the weights come from the unbiased scores, as for text). An image mask per chunk row
   travels with the chunk to GPU 1 (layers 20-39) in the stage hand-off.
3. **Engram** (layers 1 and 14; `engram.py`): image cells are DEAD. They get no Engram contribution (gate 0, no row
   reads), and no n-gram reaches across them: text after an image hashes with pad ids where the lookback hits the span.
   `engram_hash` maps negative (image) ids to DEAD.
4. **Positions**: plain 1D, one per cell (no M-RoPE). Nothing to add.
5. **Long-prompt skip, prefix forking, snapshots, adaptive exchange**: no change expected. The skip is exact, and image
   rows are rows; routing counts include image tokens. Covered by the checks in 2e.
6. **Drafters**: DSpark sees image cells as the image token's embedding plus the real target hidden states, so only
   acceptance can suffer. Lookup drafts must never propose a negative id.

### 2e. Verification

* Encoder against an fp32 PyTorch run of the base repo's `vision.py` (CPU), several sizes including the cap and an
  extreme aspect: cosine per token, like the Qwen numbers in VISION.md. The aligner and learned embeddings are checked the
  same way.
* Function: the base repo's `inference/examples/images` (carrots, corn) described correctly; OCR of a screenshot; a
  screenshot in a tool result (agent loop); the same image resent in a follow-up turn reuses its cells.
* Sanity: the same prompt with `exp_probs_b` in place of `_vl` should read worse; if not, the mask is not reaching
  the router.
* Text-only output identical to master token for token, with and without the encoder loaded (greedy, DSpark on);
  decode speed within noise after the workspace's cache cut.
* VRAM: a capped image during a 120K session with a Plex transcode on GPU 0 stays under budget.

### 2f. Docs

VISION.md (the DeepSeek tower, its grid and cap, measurements), DEEPSEEK.md and README (remove "vision not ported"),
both configs' notes (the IQ2_XS benchmark row).

## Order

1 first (small, useful without images). Then 2a -> 2c (encoder, verified alone against PyTorch) -> 2b -> 2d -> 2e -> 2f.
The fused attention kernel goes in with 2c and lands for Qwen's encoder too.

## Open questions

* The encoder workspace: from GPU 0's cache (planned), or GPU 1's? GPU 1 runs the verify rounds; GPU 0 also serves Plex.
* `exp_probs_b_vl` for the abliterated model: shared with ds-xs (the ablation did not touch the router). Assumed, not
  measured.

## Status (2026-10-28, branch ds-vision)

* Item 0 kept: the 4-shard pin; only the mmproj and `exp_probs_b_vl.f32` come from `a419d74c`.
* Item 1 done: `templates/deepseek-v4.1.jinja` through `"chat_template_file"`; 20/21 prompts identical to
  `encoding.py` against 0/21 for the GGUF's (which also lacked the BOS token).
* Item 2 done: encoder, server, engine, docs (`docs/VISION.md`, measurements there). Encoder workspace on GPU 0
  (252 MiB, fixed). ds-xs-abl shares the files (`"mmproj_from"`), not run on the GPUs.
* The fused attention kernel of 2c was built and measured 2.4x slower than the cuBLAS path: not kept.
