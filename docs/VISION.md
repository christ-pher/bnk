# Vision: images in chat requests

The Qwen3.8 models (Flash-Next and the 27B) and DeepSeek-V4.1-Flash were trained with a vision encoder; their GGUF
repos ship it as a separate `mmproj` file (projectors `qwen3vl_merger` and `deepseek41v`). With it loaded, the server takes images in OpenAI
`image_url` parts and Anthropic `image` blocks, also inside `tool_result`s (a screenshot tool), and the model
reads them: screenshots, mockups, diagrams, photos, documents.

## Using it

A config whose `source` names an `mmproj` gets vision with no flag: `./run.sh iq3_s` loads it when it is
downloaded (`./models.sh download iq3_s` fetches it with the model). `BNK_VISION=0` serves text only.

| Config | Encoder |
|---|---|
| Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (ISTA-DASLab repo) |
| Qwen3.8-Flash-Next-FP4-Local | the same file (`"mmproj_from"`: the IQ3_S config) |
| Qwen3.8-27B-GSQ-RCO-IQ3_S | `mmproj-Qwen3.8-27B-BF16.gguf` (ISTA-DASLab repo) |
| Qwen3.8-27B-Q4, -Q8_0, -NVFP4 | `mmproj-BF16.gguf` (unsloth repo; NVFP4 through `"mmproj_from"`) |
| CYBER-FROST-3.8 | none published |
| DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS (ds-xs) | `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` and `exp_probs_b_vl.f32` (ISTA-DASLab repo at `a419d74c`: `"mmproj_revision"`, `"mmproj_extra"`) |
| Huihui-...-abliterated-IQ2_XS (ds-xs-abl) | the same files (`"mmproj_from"`: ds-xs; the abliteration leaves the router alone) |

```json
"messages": [{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "text", "text": "Build this page."}]}]
```

Images come as base64 data URLs or http(s) URLs (`BNK_IMAGE_URLS=0` refuses URLs). Each is scaled to sides that
are multiples of 32 pixels (Qwen2-VL's `smart_resize`) and costs one token per 32x32 block: 1280x800 is 1,000
tokens, a 1920x1080 screenshot 2,040. `--image-max-tokens` (default 4096, `BNK_IMAGE_MAX_TOKENS`) scales larger
images down; small text in a big screenshot reads better cropped than shrunk.

## How it works

**Server** (`serve/vision.py`): the chat template renders each image as `<|vision_start|><|image_pad|><|vision_end|>`;
after tokenizing, each pad becomes a run of placeholder tokens, one per merged 32x32 block, all holding the image's
id: `-1 - hash(pixels)`. The resized pixels go to `~/.cache/bnk/images/<hash>-WxH.rgb` (4 GiB kept, oldest
first out), and the request lists `{id, path, h, w}` per image. Because an image's cells carry its id, prefix
reuse, parking, forking and snapshots compare images like tokens: a follow-up turn that resends the same
screenshot reuses its cells; a different image under the same text does not.

**Encoder** (`src/vision/vision.cu`): the ViT (27 layers x 1152, 16 heads, 2D rotary positions over patch rows and
columns, a learned 48x48 position table resized to the image) and the merger (each 2x2 patch block -> one
embedding of the model's width), on cuBLAS fp16 GEMMs with fp32 accumulation and residual. The weights stay in
page-locked host RAM as fp16 (850-870 MiB) and stream to the GPU one layer ahead, so an idle encoder holds no VRAM;
its activations are borrowed from the first GPU's expert cache for the length of an encode and given back after
(the cache refills to 100% on IQ3_S). Encoded images stay in a host cache (`--vision-cache-gib`, 2 GiB).

**Language model** (`src/engine`, `src/dense`): the placeholder cells embed as the vocabulary's image token
(also what the PLE n-gram table hashes for them, as the reference does), then the encoder's rows replace their
embeddings in the prompt chunk. Positions are M-RoPE, interleaved (`rope.dimension_sections` 11/11/10 over the 32
rotary pairs: pair p takes time, row or column by p mod 3): text has all three equal; an image's cells share its
first position p as time and take p + row and p + column; the text after continues at p + max(rows, columns). Each
conversation slot keeps a table of these positions per cell (`Seq::rpos`, 3 MiB at 256K), filled when a prompt
with images is read (and reset to the plain cell index after); attention, the QSA indexer queries and the pooled
block keys rotate with it. Without `--mmproj` no table exists and every kernel takes the cell index as before.
Image cells always go through prompt chunks (never the decode-window path for short tails).

## Measurements (two V100s, 2026-10-20)

* Encoder against an fp32 numpy reference of the HF tower (640x480, 300 tokens): cosine per token mean 0.9996,
  worst 0.957 (a few sensitive tokens; llama.cpp's CPU encoder: 0.9974 / 0.889). 256x192 against llama.cpp: worst
  0.999. `tests/test_vision` encodes an image file for such comparisons.
* Positions match llama.cpp's mtmd (`mtmd_image_tokens_get_decoder_pos`).
* Text-only output is identical to master token for token (IQ3_S and 27B NVFP4, greedy, MTP / DFlash2 on), with or
  without the encoder loaded; decode speed unchanged.
* Encode time (alone on GPU 0): 640x480 80 ms, 1280x800 0.25 s, 1920x1080 1.0 s, 2048x2048 (4,096 tokens) 4.0 s. At 1080p the
  attention takes 0.8 s of it (scores through memory: cuBLAS QK^T, a softmax pass, PV); a fused flash-attention
  kernel is the next step for screenshot-heavy agent loops.
* Turns (IQ3_S, served): a 1,047-token prompt with a 1280x800 image read in 1.25 s cold; the follow-up turn reused
  1,040 tokens and read in 0.21 s. Decoding after an image: 107-127 tok/s with the MTP drafter (3.4 tokens/round).

## DeepSeek-V4.1-Flash

The base model (deepseek-ai/DeepSeek-V4.1-Flash-1017, `image-text-to-text`) has its own encoder and its own way of
placing an image in the prompt; its reference is the repository's `inference/vision.py` and `image_processor.py`.
ISTA's mmproj holds the checkpoint's tensors bit for bit (all 263 compared); `exp_probs_b_vl.f32` is the checkpoint's
`gate.bias_vl` of the 40 layers (the file of their later layout embeds it; ours is the earlier one, docs/DEEPSEEK-UPDATE-PLAN.md).

**Images** (`serve/vision.py` `deepseek_grid`, identical to `plan_image_grid` on 20,011 sizes): 14-pixel patches, one
cell per 3x3 of them; an image under 544x544 pixels of area is scaled up, a larger one down until its span fits 1,024
cells; it is letterboxed in grey to whole patches, not stretched. The span the model reads: a start cell, each row of
the grid's cells followed by a newline cell, an end cell (`rows * (columns + 1) + 2`). A 1920x1080 screenshot is
23 x 41 cells, 968 tokens; 1280x800, 20 x 31, 642. The template (`templates/deepseek-v4.1.jinja`) renders an image as
`<｜deepseek_image｜>`, also inside tool results; the server lists the images in the order the template renders
them (parallel tool results in call order, as DeepSeek's encoder sorts them).

**Encoder** (`src/vision`, projector `deepseek41v`): a 32-layer ViT (1024 wide, 16 heads, RMSNorm, SwiGLU 2816,
2D rotary positions over patch rows and columns, no position table), the aligner (every 3x3 block of patches, the
grid zero-padded to whole blocks, unfolded as torch's `unfold` -> Linear 9216 -> 5120 -> GELU -> Linear), then the
start / newline / end embeddings around the rows. Weights fp16 in page-locked host RAM (926 MiB), streamed in parts
of at most ~26 MB. Its scratch is fixed: 252 MiB on GPU 0, charged to the budget before the expert cache sizes itself
(the DeepSeek engine's cache cannot lend VRAM for an encode the way the Qwen engine's does); the QKV and MLP run in
row blocks and the attention scores take 32 MiB a pass, so the largest image fits.

**Language model** (`src/deepseek`): image cells take the encoder's rows in the prompt chunk (from the image token's
row, overwritten); they pick their experts with `exp_probs_b_vl` instead of `exp_probs_b`, on both stages (the bias
only chooses; the weights come from the unbiased scores); they get no Engram contribution, and no n-gram reaches across
them (negative ids hash as dead, as `engram.py` masks them). Positions stay one per cell (no M-RoPE). Image cells
always go through prompt groups; a decode window refuses them. DSpark drafts from the target layers' rows, which
include the image's real rows; lookup drafts stop before image cells (`generate.cpp`). `BNK_DS_VL_BIAS=0` routes
image cells with the text bias (for checks).

### Measurements (two V100s, 2026-10-28)

* Encoder against an fp32 numpy port of `vision.py` (`tools/ds_vision_ref.py`, the reference rows cached with
  `--cache`): cosine per cell, mean / worst: 630x476 0.99993 / 0.9935, 1036x714 0.99997 / 0.9909, 658x462
  0.99999 / 0.9997, 308x4004 1.00000 / 0.99995, 1708x966 0.99996 / 0.9862.
* Encode time: 194 cells 82 ms, 444 cells 300 ms, 866 cells 758 ms, 968 cells (1080p) 1.45 s.
* A fused tensor-core attention (WMMA, online softmax, 64 rows a block) was tried for both encoders: 2.4x slower
  than the cuBLAS path at every size (one block an SM for its 66-80 KB of shared memory, a sync per key tile) and
  less accurate (worst cell 0.79 at 1080p). Not kept.
* Served (ds-xs, greedy): DeepSeek's own example (carrots and corn, the question in Chinese) answered correctly; a
  1080p screenshot's 12 code lines transcribed exactly (12/12), its rectangle's color right; a screenshot in a tool
  result read; a small (120x90) and a tall (300x4000) image read; a follow-up turn resending the image reused 474 of
  485 tokens (0.23 s read). Prompts with an image read at 215-450 tok/s including the encode (996 tokens with a 1080p
  screenshot: 2.2 s).
* The routing bias reaches both stages (`BNK_DS_VL_TRACE=1`: each image group routed with both biases in one pass):
  with `exp_probs_b_vl` 590-627 of 633 image rows pick other experts in layers 0-6, 186-270 in layers 30-38, 446 in
  layer 39; text rows never.
* Text without images: the same kernels as before (no mask, the text bias); the workspace took nothing from the expert
  cache here (GPU 0's cache is capped at 20 GiB with 21.6 GiB free). Decode 27-31 tok/s with and without the encoder
  loaded (DeepSeek's greedy output differs between runs anyway: which experts the CPU computes changes the rounding).

## Limits

* Video is not supported (`<|video_pad|>`); images only.
* DeepSeek: at most 1,024 cells an image (the model's training cap); larger screenshots are scaled down, so small
  text reads better cropped.
* The drafter (MTP / DFlash2) sees image cells as the image token at their cell positions, not the encoder's rows:
  only its acceptance can suffer, never the output.
* Image ids are 30-bit hashes: two different images colliding in one server's lifetime would share a cache entry.
