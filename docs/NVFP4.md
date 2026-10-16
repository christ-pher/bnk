# NVFP4 / MXFP8 models on the V100s (branch `nvfp4-volta`, started 2026-10-16)

Goal: serve NVIDIA's `nvidia/Qwen3.8-Flash-Next-FP4-Local` (ModelOpt NVFP4 experts, MXFP8 dense, quantization-aware
distilled: GPQA 91.5 vs 92.2 for BF16) on the two V100s, and more generally load FP4 / FP8 weight formats the cards'
silicon does not know.

## Can a V100 be made to run FP8 / FP4 arithmetic? No, and it does not need to

* The tensor cores of a V100 (sm_70) are fixed-function units for one shape of product: fp16 x fp16 -> fp32
  (`mma.m8n8k4`). INT8 tensor-core products came with Turing (sm_75), BF16 / TF32 with Ampere, FP8 with Hopper and
  Ada (sm_89 / 90), FP4 with Blackwell (sm_100 / 120). These are datapaths etched in the SM, not microcode: no
  VBIOS, firmware or driver can add them, and `ptxas` refuses the instructions for sm_70. CUDA 12.8 is also the last
  toolkit that targets sm_70 at all.
* What the newer cards' FP4 / FP8 *GEMMs* do is multiply low-precision weights by low-precision activations
  natively. What actually matters for a model file is the **weight encoding**, and that is just bits: e2m1 values
  (`{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with a sign) with one e4m3 scale per 16 weights (NVFP4), or e4m3 values with one
  power-of-two exponent per 32 weights (MXFP8). Decoding them to fp16 in registers is cheaper than decoding the
  IQ2_S / IQ3_S codebooks bnk already runs, and the fp16 products on the V100 are *more* precise than the native
  W4A4 the model was trained for (the activations stay int8 or fp16 instead of being rounded to e2m1).
* So the software route is the real one: bnk treats NVFP4 and MXFP8 as two more weight formats in its existing
  quantized GEMV / expert kernels (unpack a 32-weight sub-block to int8 x per-16 scales or to fp16, feed `dp4a` or
  the fp16 tensor cores). Nothing is emulated; the arithmetic is fp16 / int8, and the model's weights are used bit
  for bit.

## The checkpoint (ModelOpt v0.47, config.json `quantization_config`)

| Tensors | Format | Bytes | Notes |
|---|---|---|---|
| routed experts (48 x 512 x gate/up/down) | NVFP4, group 16, `weight_scale_encoding: csf` | 59.9 GiB | u8 codes [out, in/2]; scales in the csf form below; fp32 `weight_scale_2` per matrix (= amax / (6 x 448)); `input_scale` fp32 (activations, unused by us) |
| linear-attention and attention projections, shared experts, indexer | MXFP8, group 32 | 2.8 GiB | e4m3 weights, u8 UE8M0 exponent per 32 weights |
| hyper-connections, routers, norms, embeddings, lm_head, DeltaNet small tensors | BF16 / F32 | 3.8 GiB | as in the BF16 release |
| PLE n-gram table (layer 1, 320M rows x 160) | NVFP4, group 16, dense e4m3 scales `[rows, 10]`, one `weight_scale_2` | 26.8 GiB | 128 shards; 160 is not a multiple of ggml's 64-wide NVFP4 block |
| MTP head | NVFP4 experts (W4A16, dense scales), BF16 rest | 1.5 GiB | bnk drafts with its own MTP file |
| vision tower | MXFP8 / NVFP4 / BF16 | 0.3 GiB | not used |

**csf scales** (verified on a real expert: scale bytes decode to sane e4m3 values, `6 x 448 x weight_scale_2` equals
the implied max |w|, and the e2m1 code histogram is symmetric in sign): `weight_scale.nvfp4_csf_fixed` u8
`[rows/16, 16 + 16 x (C/16)/2]`: per 16-row slab, 16 row bases then each row's 4-bit codes (low nibble first), scale
byte = base + code; `weight_scale.nvfp4_csf_exceptions` u32 `[n]`: `(row x C/16 + col) | byte << 24`, the byte
replaces the decoded one. The Qwen checkpoint stores one expert per tensor, so there is no `csf_starts`.

## GGUF layout bnk reads (= what llama.cpp writes)

llama.cpp (`conversion/base.py`, pinned commit `f3363593`) repacks ModelOpt NVFP4 bit for bit into ggml's
`GGML_TYPE_NVFP4` (type 40: 64 weights = 4 UE4M3 scale bytes + 32 code bytes, 4.5 bits per weight) and keeps the
per-matrix fp32 scale as a side tensor: `blk.N.ffn_{gate,up,down}_exps.weight` (NVFP4 `[512, rows, cols]`) with
`blk.N.ffn_{gate,up,down}_exps.scale` (F32 `[512]`) and `.input_scale` (ignored). The weight is
`code x ue4m3_half(scale_byte) x scale[e]`, with ggml's convention of doubled e2m1 values and halved scales.

`tools/convert_nvfp4.py` drives that converter (its own venv with CPU torch: `~/.cache/bnk/convert-venv`, the
checkout in `~/.cache/bnk/llama.cpp`), adding what the checkpoint needs: the csf decode into dense e4m3
`weight_scale` tensors, the MXFP8 dequantization (e4m3 x 2^(e - 127)) stored as **Q8_0** (`--fp8 q8_0`; BF16 with
`--fp8 bf16`), the PLE table left out (`--no-ple`), the MTP head left out. Q8_0 is not bit-exact for MXFP8 (int8
relative to a block max of 32 instead of 3 mantissa bits at every magnitude): weights below max/16 of their block
lose precision, the rest gain. A native MXFP8 format in bnk (type 43, 32 e4m3 bytes + 1 exponent per sub-block,
fp16 unpack) is the exact follow-up if `bnk nll` shows a difference.

## Engine support

* Done: `QT_NVFP4` in `kernels/quant.cuh` (raw ggml blocks), `F_NVFP4` R layout (`rfmt.cpp` / `rfmt_impl.h`:
  16 code bytes + 2 scale bytes per 32-weight sub-block, same nibble order as IQ4_NL) and its device unpack
  (`rfmt.cuh`), the dispatch macros, 64-column alignment in `upload_matrix`. `tests/make_quant_gguf.py` writes a
  GGUF of random NVFP4 blocks; `build/test_dequant` on it is bit-exact against ggml's `dequantize_row_nvfp4` in
  both layouts, GEMV error 5.5e-3 (Q8_0 on the same test: 5.3e-3). The CPU expert path takes NVFP4 through
  ggml-cpu's `ggml_vec_dot_nvfp4_q8_0` (AVX2 version present).
* The per-expert scales: `scale_up[e] x scale_down[e]` is folded into the routing weight of expert e (exact: both
  are linear), `scale_gate[e]` is applied to the gate rows before the silu (the only place it cannot move). Sites:
  the routing-weight kernels (decode, prompt, MTP), the activation kernels (decode, prompt list, tensor-core
  prompt), the CPU expert pool.
* Memory on two 32 GB cards: 59.9 GiB of experts + ~6 GiB dense (Q8_0 + BF16) + KV / prompt arena: ~75-80% of the
  experts resident, the rest computed by the CPU (as CYBER-FROST Q5_K_M at 65%: 88 tok/s with MTP). The IQ3_S
  release keeps every expert resident (46.8 GiB) and decodes at 105-120 tok/s; expect NVFP4 between the two, at
  better quality.

## Status (2026-10-16)

* The checkpoint converts in ~15 minutes (shards 1-15 of the Hub snapshot in `~/.cache/huggingface`, output
  `~/.cache/bnk/gguf/Qwen3.8-Flash-Next-FP4-Local.gguf`, 75.2 GB: experts NVFP4 63.3 GiB + scales, dense Q8_0 2.9
  GiB, BF16 3.6 GiB; the converter holds ~70 GB of RAM). `test_dequant` on its real tensors is bit-exact in both
  layouts; `test_moe_tc` on layer 0 (scales applied) is within 3e-4 of the fp32 reference.
* It runs on the two V100s through `configs/Qwen3.8-Flash-Next-FP4-Local.json` (alias `nvfp4`) with
  `BNK_PLE_GGUF=<IQ3_S first shard>` for the n-gram table: 74% of the experts resident per card (24.5 / 22.7 GiB
  caches), greedy output coherent, the first 8 tokens of the 4K follow-up equal to the IQ3_S release's, drafts
  accepted 37/48. First run, uniform expert ranking: 56 tok/s decode with 22% expert misses on the CPU, 4,700-token
  cold read 481 tok/s (stage 0 streams the non-resident experts). With the routing counts of that same prompt
  (`--counts`, a second pass): **98.8 tok/s**, 2.8% misses, drafts accepted 44/49 (the IQ3_S release, every expert
  resident: 105-120). A ranking learned on one prompt flatters the miss rate; the server's counts over real traffic
  are the honest number.
* Two bugs found on the way, both fixed: the global scales cannot ride on the routing weights (the prompt path
  stores the unscaled fp16 products: overflow), so they are applied inside the expert kernels; and the loader
  only converted BF16 to F16 for widths that are multiples of 32, which left the PLE conv taps (4 per row) as
  BF16 for a kernel that reads raw fp16: the n-gram injection came out ~23x too large (the IQ3_S release stores
  that tensor as F16, so it never showed). Found with the layer dumps (`BNK_DUMP_DIR` / `BNK_DUMP_LAYER`) compared
  between the two models: layer 0 agreed to 0.97-0.99 correlation everywhere, layer 1 did not.

* **The n-gram table in NVFP4** (later the same day): `tools/convert_ple_nvfp4.py` packs the 128 table shards into
  `~/.cache/bnk/gguf/Qwen3.8-Flash-Next-FP4-Local-ple.gguf` (28.8 GB, 2 minutes, numpy only) as bnk's
  `GGML_TYPE_NVFP4_16` (type 43: one UE4M3 byte + 8 code bytes per 16 weights, added to the vendored ggml with its
  reference dequantizer), plus `per_layer_token_embd.scale` (the checkpoint's `weight_scale_2`, applied by the
  gather through `QMat::scale`). Self-check: the packed rows decode to the checkpoint's values exactly;
  `test_dequant` (`BNK_TEST_PLE=1`) bit-exact against ggml in both layouts; the table correlates 0.992 with the
  base model's (the distillation barely touched it). The config's `ple_gguf` makes `./run.sh nvfp4` pick it up.
  Run with the ranked cache: 93.6 tok/s, drafts 41/49, 0.17% misses; the first 37 tokens equal the borrowed-table
  run's. Nothing from the IQ3_S release is used any more.

## Plan

1. ~~NVFP4 weight format in the kernels, bit-exact test~~ (done).
2. ~~Convert the checkpoint, validate the GGUF~~ (done).
3. ~~Loader and kernels: the per-expert scales~~ (done).
4. ~~First run with the IQ3_S release's table~~ (done). Next: a server run to learn the routing counts, then the
   LocalMaxxing sweep (`/opt/benchmarks/bnk_bench.py`) and `bnk nll` against the IQ3_S release for the quality
   difference; re-tune `--draft` / `--min-p` for this model.
5. ~~The PLE table in NVFP4~~ (done, above). The checkpoint shards still in the Hugging Face cache (15 to 21,
   ~32 GB) are no longer needed.
6. Optional: native MXFP8 type for the dense weights (Q8_0 doubles their bytes: 2.9 vs the release's ~2 GiB per
   card, and the output head is BF16 1.2 GiB, Q6_K in the release); the MTP head from this checkpoint (NVFP4
   experts) in place of bnk's Q2_0 one built from the base model.
